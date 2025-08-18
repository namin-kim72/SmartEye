# 🚀 SmartEye  
> 실시간 카메라 기반 보조 시스템  
> 보행 안전 · 텍스트 읽기 · 음식 인식 기능을 하나로 통합  

---
## 📂 Project Structure

```text
SmartEye/
│
├─ Common/                  # 공용 모듈
│   ├─ capture.py
│   ├─ img_processing.py
│   └─ config.py
│
├─ MealMode/                # 식사 모드 (EfficientDet 기반 음식 인식)
│   ├─ foodMode.py
│   ├─ config.py
│   └─ model/
│       ├─ food_effi_lite0_200ep_int8.tflite
│       └─ labels.txt
│
├─ ReadMode/                # 읽기 모드 (OCR + Gesture + TTS)
│   ├─ ReadMode.py
│   └─ config.py
│
├─ WalkingMode/             # 보행 모드 (객체 감지 + 거리 추정 + 위험 알림)
│   ├─ WalkingMode.py
│   ├─ deep_learning.py
│   ├─ risk.py
│   ├─ notifier.py
│   ├─ config.py
│   └─ model/
│       ├─ ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
│       └─ coco_labels.txt
│
└─ main.py                  # 메인 실행기 (모드 선택 메뉴)
```

## 메뉴
'''text
=== SmartEye 메인 ===
  1. 보행모드 (WalkingMode)
  2. 읽기모드 (ReadMode)
  3. 식사모드 (MealMode)
  q. 종료
'''

1: 보행모드

객체 감지 (사람, 차량, 신호등 등)

거리 추정 + 위험 판단

진동/사운드 알림

2: 읽기모드

손 제스처(주먹) → 캡처 트리거

Google Vision OCR로 텍스트 추출

음성(TTS)으로 읽어주기

3: 식사모드

EfficientDet 기반 음식 탐지

라벨 표시 및 FPS 출력
