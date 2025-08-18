import cv2
import mediapipe as mp
import time
from picamera2 import Picamera2
import os
from google.cloud import vision
import pyttsx3

# --- 설정 및 초기화 ---
# Google Cloud
GOOGLE_VISION_KEY_PATH = '/home/pi/readmode-7aa2558a63b1.json'
IMAGE_FILE_PATH = "/home/pi/captured_image_for_ocr.jpg"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_VISION_KEY_PATH

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# pyttsx3 엔진 초기화
tts_engine = pyttsx3.init()

# 1. 요청하신 대로 Picamera2 설정을 RGB 형식으로 변경합니다.
picam2 = Picamera2()
capture_config = picam2.create_still_configuration(
    main={"size": (width, height), "format": "RGB888"},
    lores=None,
    raw=None,
    colour_space=libcamera.ColorSpace.Raw(),
    buffer_count=6,
    queue=True
)
picam2.configure(capture_config)
picam2.start()


# --- 핵심 기능 함수 (수정 없음) ---
def is_fist(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [6, 10, 14, 18]
    for i in range(4):
        tip = hand_landmarks.landmark[finger_tips[i]]
        mcp = hand_landmarks.landmark[finger_mcp[i]]
        if tip.y > mcp.y: return False
    return True


def detect_text(image_path):
    print("글자를 찾는 중입니다...")
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as image_file: content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    if response.full_text_annotation: return response.full_text_annotation.text
    return "인식된 글자가 없습니다."


def speak(engine, text):
    print("음성으로 변환하여 읽어드립니다...")
    engine.setProperty('rate', 170)
    engine.say(text)
    engine.runAndWait()


# --- 메인 프로그램 ---
def main():
    fist_detected_time = None
    capture_triggered = False

    while True:
        # 2. 이제 frame은 RGB 형식입니다.
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.rotate(frame_rgb,cv2.ROTATE_90_COUNTERCLOCKWISE)
        # 3. MediaPipe는 RGB 형식에서 더 잘 동작하므로 그대로 사용합니다.
        results = hands.process(frame_bgr)

        if results.multi_hand_landmarks:
            print("found hand")
            for hand_landmarks in results.multi_hand_landmarks:
                # BGR 이미지 위에 랜드마크를 그립니다.
                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if is_fist(hand_landmarks):
                    if fist_detected_time is None: fist_detected_time = time.time()
                    if time.time() - fist_detected_time > 3.0:
                        cv2.putText(frame_bgr, "CAPTURING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        capture_triggered = True
                        break
                    else:
                        countdown = 3 - int(time.time() - fist_detected_time)
                        cv2.putText(frame_bgr, f"Fist detected... {countdown}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
                else:
                    fist_detected_time = None
        else:
            fist_detected_time = None

        # 5. 최종적으로 색상이 변환된 BGR 이미지를 화면에 보여줍니다.
        cv2.imshow('Gesture Cam', frame_bgr)

        if (cv2.waitKey(5) & 0xFF == ord('q')) or capture_triggered:
            break

    cv2.destroyAllWindows()

    if capture_triggered:
        print("\n--- OCR 및 음성 출력 실행 ---")
        # 고해상도 이미지를 촬영할 때도 RGB로 받아서 저장합니다.
        config = picam2.create_still_configuration(main={"format": "RGB888"})
        picam2.switch_mode(config)
        picam2.capture_file(IMAGE_FILE_PATH)
        print(f"고해상도 사진 저장 완료: {IMAGE_FILE_PATH}")

        text = detect_text(IMAGE_FILE_PATH)
        print(f"인식된 텍스트:\n{text}")
        speak(tts_engine, text)

    picam2.stop()
    hands.close()
    print("프로그램 종료.")


if __name__ == '__main__':
    main()
