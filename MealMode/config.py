# -*- coding: utf-8 -*-
# 식사 파일에서 공통으로 관리할 변수들

# 식사 모델 경로
MEAL_MODEL_PATH = "./model/food_effi_lite0_200ep_int8.tflite"
MEAL_LABEL_PATH = "./model/labels.txt"

# 추론 관련
SCORE_THRESHOLD = 0.15        # 0.15~0.5 사이에서 필요하면 조절
DEDUP_DISTANCE = 15           # 같은 클래스에서 중심 간격이 이 값 이하면 중복으로 간주

# 화면/표시
WINDOW_TITLE = "Meal: EfficientDet (EdgeTPU)"
SHOW_WINDOW = True            # False면 창 띄우지 않고 내부만 동작
SHOW_FPS = True               # 좌상단 FPS/지연 표시
FONT_COLOR = (255, 255, 0)    # 노란색 (BGR)
BOX_COLOR = (0, 255, 255)     # 노란색 계열 (BGR)
DISPLAY_SIZE = (480, 640)
# 카메라
# Common.config에 있는 해상도를 그대로 쓰도록 권장.
USE_PORTRAIT = False          # 세로 비율 필요시 True로 두고 width/height를 뒤집어 적용
