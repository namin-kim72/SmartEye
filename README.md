# 🚀 SmartEye  
> 실시간 카메라 기반 보조 시스템  
> 보행 안전 · 텍스트 읽기 · 음식 인식 기능을 하나로 통합  

---

## 📂 Project Structure

SmartEye/
│
├─ Common/ # 공용 모듈
│ ├─ capture.py
│ ├─ img_processing.py
│ └─ config.py
│
├─ MealMode/ # 식사 모드 (EfficientDet 기반 음식 인식)
│ ├─ foodMode.py
│ ├─ config.py
│ └─ model/
│ ├─ food_effi_lite0_200ep_int8.tflite
│ └─ labels.txt
│
├─ ReadMode/ # 읽기 모드 (OCR + Gesture + TTS)
│ ├─ ReadMode.py
│ ├─ config.py
│
├─ WalkingMode/ # 보행 모드 (객체 감지 + 거리 추정 + 위험 알림)
│ ├─ WalkingMode.py
│ ├─ deep_learning.py
│ ├─ risk.py
│ ├─ notifier.py
│ ├─ config.py
│ └─ model/
│ ├─ ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
│ └─ coco_labels.txt
│
└─ main.py # 메인 실행기 (모드 선택 메뉴)
