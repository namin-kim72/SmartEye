# WalkingMode/config.py

# 모델/라벨 경로 (모드 폴더 기준 상대경로)
WALKING_MODEL_PATH = "model/mobilenet_coco/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
WALKING_LABEL_PATH = "model/mobilenet_coco/coco_labels.txt"

# 화면 표시 여부
SHOW_WINDOW = True
SHOW_FPS = True

# 추론 관련 설정
SCORE_THRESHOLD = 0.5
DEDUP_DISTANCE = 0.2

# COCO 클래스 ID (사람, 자전거, 차 등)
TARGET_CLASS_IDS = [0, 1, 2, 3, 5, 7, 9]

# 위험 거리 임계(mm) — class_id별
DANGER_THRESHOLDS = {
    0: 1000,
    1: 1400,
    2: 1500,
    3: 1400,
    5: 1500,
    7: 1600,
    9: 1800,
    "default": 1200
}

# 추론/표시 옵션
CONF_THRESHOLD = 0.4
WINDOW_TITLE = "Walking: RoadFriend"
DISPLAY_SIZE = (480, 640)

# FPS 제어 (프레임 초당 계산)
FPS_LIMIT = 30

# 디버그 옵션
DEBUG_MODE = False
LOG_DETECTED_OBJECTS = True

# 카메라 관련 추가 설정
CAMERA_RESOLUTION = (640, 480)
CAMERA_ROTATE_CCW_90 = True
CAMERA_FPS = 30

# 박스 색상
BOX_COLOR = (255, 0, 0)
FONT_COLOR = (255, 255, 255)

# 각도 센서 관련 설정
USE_ANGLE_SENSOR = True            # 각도 센서 사용 여부
