# Common/capture.py

from picamera2 import Picamera2
import libcamera
import cv2
import threading
import queue

from Common.config import (
    CAMERA_RESOLUTION, CAMERA_FORMAT, PICAMERA_BUFFER_COUNT,
    USE_RAW_COLORSPACE, CAMERA_ROTATE_CCW_90
)

class CameraManager:
    def __init__(self, bus, still_capture_queue):
        self.bus = bus
        self.still_capture_queue = still_capture_queue
        self.picam2 = None
        self.running = False
        self.thread = None

        # 이곳에서 하나의 Picamera2 객체만 생성합니다.
        self.picam2 = Picamera2()

        # 이제 이 하나의 객체를 사용하여 설정을 만듭니다.
        self.preview_config = self.picam2.create_preview_configuration(
            main={"format": CAMERA_FORMAT, "size": CAMERA_RESOLUTION},
            buffer_count=PICAMERA_BUFFER_COUNT,
            colour_space=libcamera.ColorSpace.Sycc() if not USE_RAW_COLORSPACE else libcamera.ColorSpace.Raw()
        )
        self.still_config = self.picam2.create_still_configuration(main={"format": "RGB888"})

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.thread.join()

    def _capture_loop(self):
        try:
            # 단일 카메라 인스턴스를 설정하고 시작합니다.
            self.picam2.configure(self.preview_config)
            self.picam2.start()
            print("[CAMERA] 카메라 캡처 스레드 시작")

            while self.running:
                try:
                    capture_path = self.still_capture_queue.get_nowait()
                    self._do_still_capture(capture_path)
                except queue.Empty:
                    pass
                frame = self.picam2.capture_array()
                if CAMERA_ROTATE_CCW_90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                self.bus.publish(frame)
        except Exception as e:
            print(f"[ERROR] 카메라 캡처 스레드 오류: {e}")
        finally:
            if self.picam2:
                self.picam2.stop()
                self.picam2.close()
            print("[CAMERA] 카메라 캡처 스레드 종료")

    def _do_still_capture(self, file_path):
        try:
            self.picam2.switch_mode_and_capture_file(self.still_config, file_path)
            print(f"[CAMERA] 고해상도 이미지 저장: {file_path}")
        except Exception as e:
            print(f"[ERROR] 고해상도 캡처 실패: {e}")

    def request_still_capture(self, file_path):
        self.still_capture_queue.put(file_path)
