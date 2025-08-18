# WalkingMode/config.py
# 보행 모드 전용 설정

# 모델/라벨 경로 (모드 폴더 기준 상대경로)
WALKING_MODEL_PATH = "model/mobilenet_coco/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
WALKING_LABEL_PATH = "model/mobilenet_coco/coco_labels.txt"

# 감지할 COCO 클래스 ID
TARGET_CLASS_IDS = [0, 1, 2, 3, 5, 7, 9]  # person,bicycle,car,motorcycle,bus,truck,traffic light

# 위험 거리 임계(mm) — class_id별
DANGER_THRESHOLDS = {
    0: 1000,   # person
    1: 1400,   # bicycle
    2: 1500,   # car
    3: 1400,   # motorcycle
    5: 1500,   # bus
    7: 1600,   # truck
    9: 1800,   # traffic light
    "default": 1200
}

# 추론/표시 옵션
CONF_THRESHOLD = 0.4
WINDOW_TITLE = "Walking: RoadFriend"
DISPLAY_SIZE = (480, 640)  # imshow 리사이즈
