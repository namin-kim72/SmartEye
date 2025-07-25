# capture.py
from picamera2 import Picamera2
import cv2

# Picamera2 설정
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (320, 240)}
))
picam2.start()

def Capture():
    try:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return frame_bgr
    except Exception as e:
        print(f"[ERROR] Capture 실패: {e}")
        return None
