import cv2, time, os
import mediapipe as mp
from google.cloud import vision
import pyttsx3

# --- 설정 ---
from ReadMode.config import (
    GOOGLE_VISION_KEY_PATH, IMAGE_FILE_PATH,
    FIST_HOLD_TIME, COUNTDOWN_COLOR, CAPTURE_COLOR,
    TTS_RATE, WINDOW_TITLE,
)
from Common.capture import Capture

# Google Vision API 키 환경 변수
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_VISION_KEY_PATH

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# TTS 초기화
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", TTS_RATE)

# --- 보조 함수 ---
def is_fist(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [6, 10, 14, 18]
    for i in range(4):
        if hand_landmarks.landmark[finger_tips[i]].y > hand_landmarks.landmark[finger_mcp[i]].y:
            return False
    return True

def detect_text(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as f:
        img = vision.Image(content=f.read())
    resp = client.text_detection(image=img)
    return resp.full_text_annotation.text if resp.full_text_annotation else "인식된 글자가 없습니다."

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# --- 메인 ---
def main():
    fist_detected_time = None
    capture_triggered = False

    while True:
        frame_rgb = Capture()
        if frame_rgb is None:
            continue

        # MediaPipe는 RGB로 받음
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if is_fist(hand_landmarks):
                    if fist_detected_time is None:
                        fist_detected_time = time.time()
                    if time.time() - fist_detected_time > FIST_HOLD_TIME:
                        cv2.putText(frame_rgb, "CAPTURING!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, CAPTURE_COLOR, 2)
                        capture_triggered = True
                        break
                    else:
                        countdown = int(FIST_HOLD_TIME - (time.time() - fist_detected_time))
                        cv2.putText(frame_rgb, f"Fist detected... {countdown}", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, COUNTDOWN_COLOR, 2)
                else:
                    fist_detected_time = None
        else:
            fist_detected_time = None

        cv2.imshow(WINDOW_TITLE, frame_rgb)
        if (cv2.waitKey(5) & 0xFF == ord('q')) or capture_triggered:
            break

    cv2.destroyAllWindows()

    if capture_triggered:
        # 캡처된 화면 저장 (RGB → BGR 변환 후 저장)
        cv2.imwrite(IMAGE_FILE_PATH, frame_rgb)
        text = detect_text(IMAGE_FILE_PATH)
        print(f"[OCR 결과]\n{text}")
        speak(text)

if __name__ == "__main__":
    main()
