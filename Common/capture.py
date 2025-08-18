# Common/capture.py
from picamera2 import Picamera2
import libcamera
import cv2

from Common.config import (
    CAMERA_RESOLUTION,
    CAMERA_FORMAT,
    PICAMERA_BUFFER_COUNT,
    USE_RAW_COLORSPACE,
    CAMERA_ROTATE_CCW_90,
)

# 내부 전역 변수 (아직 카메라 없음)
_picam2 = None

def _get_camera():
    """Picamera2 객체를 최초 한 번만 초기화하고 재사용."""
    global _picam2
    if _picam2 is None:
        _picam2 = Picamera2()
        colour_space = libcamera.ColorSpace.Raw() if USE_RAW_COLORSPACE else libcamera.ColorSpace.Sycc()
        preview_cfg = _picam2.create_preview_configuration(
            main={"format": CAMERA_FORMAT, "size": CAMERA_RESOLUTION},
            lores=None,
            raw=None,
            buffer_count=PICAMERA_BUFFER_COUNT,
            colour_space=colour_space,
        )
        _picam2.configure(preview_cfg)
        _picam2.start()
    return _picam2

def Capture():
    """
    표준 캡처 함수.
    반환: numpy.ndarray, 채널 순서 = RGB (Picamera2 기본)
    회전: config.CAMERA_ROTATE_CCW_90=True이면 CCW 90도 회전 적용
    """
    try:
        cam = _get_camera()
        frame = cam.capture_array()  # RGB
        if CAMERA_ROTATE_CCW_90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
    except Exception as e:
        print(f"[ERROR] Capture 실패: {e}")
        return None
