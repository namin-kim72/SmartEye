# capture.py
from picamera2 import Picamera2
import cv2

# Picamera2 설정
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640,480)}
))
picam2.start()

def Capture():
    try:
        frame = picam2.capture_array()
        return frame
    except Exception as e:
        print(f"[ERROR] Capture 실패: {e}")
        return None
