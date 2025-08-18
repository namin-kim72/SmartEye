# 공용 파일에서 공통으로 관리할 변수들

# === Camera ===
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_RESOLUTION = (CAMERA_WIDTH, CAMERA_HEIGHT)
CAMERA_FORMAT = "RGB888"             # Picamera2 main stream format
CAMERA_ROTATE_CCW_90 = True          # 공용 캡처에서 CCW 90도 회전 여부
PICAMERA_BUFFER_COUNT = 6            # ring buffer
USE_RAW_COLORSPACE = True            # libcamera.ColorSpace.Raw() 사용 여부
