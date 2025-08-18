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

---

## ⚙️ Installation
```bash
# Clone repository
git clone https://github.com/USERNAME/SmartEye.git
cd SmartEye

# 필수 패키지 설치
pip install -r requirements.txt


▶️ Usage

메인 실행기에서 모드를 선택할 수 있습니다.

python main.py

메뉴
=== SmartEye 메인 ===
  1. 보행모드 (WalkingMode)
  2. 읽기모드 (ReadMode)
  3. 식사모드 (MealMode)
  q. 종료

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

📸 Features

모듈화된 구조: Common/MealMode/ReadMode/WalkingMode 분리

Edge TPU 최적화: Coral USB Accelerator 활용

실시간 반응: 카메라 입력 기반 초저지연

멀티모드 지원: 하나의 시스템에서 다양한 보조 기능 실행

🛠️ Hardware

Raspberry Pi (Zero 2 W / CM4)

Pi Camera (CSI)

Google Coral USB Accelerator

진동 모터, 부저 등 알림 장치

👥 Contributors

Team SmartEye
