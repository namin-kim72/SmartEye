# 전체 파일에서 공통으로 관리할 변수들

# 모델 경로
MODEL_PATH = 'model/mobilenet_coco/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
LABEL_PATH = 'model/mobilenet_coco/coco_labels.txt'

# 감지할 클래스 목록
TARGET_CLASS_IDS = [0, 1, 2, 3, 5, 7, 9]  # person, bicycle, car, motorcyle, bus, truck, traffic light

# 위험 거리 기준
DANGER_THRESHOLDS = {
    0: 1000,   # person
    1: 1400,   # bicycle
    2: 1500,   # car
    3: 1400,   # motorcycle
    5: 1500,   # bus
    7: 1600,   # truck
    9: 1800,   # traffic light
    'default': 1200
}


# 카메라 관련 설정
CAMERA_RESOLUTION = (320, 240)
CAMERA_INDEX = 0
