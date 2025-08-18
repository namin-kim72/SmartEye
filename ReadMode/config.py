# 읽기 모드 파일에서 공통으로 관리할 변수들

# --- Google Cloud Vision ---
GOOGLE_VISION_KEY_PATH = "/home/pi/readmode-7aa2558a63b1.json"
IMAGE_FILE_PATH = "/home/pi/captured_image_for_ocr.jpg"

# --- 제스처 인식 ---
FIST_HOLD_TIME = 3.0   # 몇 초 동안 주먹 유지하면 캡처
COUNTDOWN_COLOR = (0, 0, 255)  # 빨강 (BGR)
CAPTURE_COLOR = (0, 255, 0)    # 초록 (BGR)

# --- TTS ---
TTS_RATE = 170          # 음성 속도 (기본=200)

# --- 창 ---
WINDOW_TITLE = "Gesture Cam (ReadMode)"
