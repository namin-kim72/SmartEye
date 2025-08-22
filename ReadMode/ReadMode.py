#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import numpy as np
import subprocess
import threading

# from picamera2 import Picamera2 # <-- 삭제
import mediapipe as mp
# from Common.capture import Capture # <-- 삭제
from google.cloud import vision
from google.cloud import texttospeech as tts

# ---- 설정 불러오기 ----
from ReadMode.config import (
    GOOGLE_VISION_KEY_PATH,
    IMAGE_FILE_PATH,
    FIST_HOLD_TIME,
    COUNTDOWN_COLOR,
    CAPTURE_COLOR,
    WINDOW_TITLE,
)
from Common.frame_bus import FrameBus

# ---- GCP 자격 ----
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_VISION_KEY_PATH

# ---- 파일 경로 ----
AUDIO_FILE_PATH = "/tmp/tts_readmode.wav"

# ---- MediaPipe Hands ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.6,
)
mp_draw = mp.solutions.drawing_utils

# ---------------------------
# 제스처 판정 유틸
# ---------------------------
def is_fist(hand_landmarks) -> bool:
    tips = [8, 12, 16, 20]
    mcps = [6, 10, 14, 18]
    for i in range(4):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[mcps[i]].y:
            return False
    return True

def is_open_hand(hand_landmarks) -> bool:
    tips = [8, 12, 16, 20]
    mcps = [6, 10, 14, 18]
    for i in range(4):
        if not (hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[mcps[i]].y):
            return False
    return True

# ---------------------------
# OCR
# ---------------------------
def detect_text(image_path: str) -> str:
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    resp = client.text_detection(image=image)
    if resp.full_text_annotation and resp.full_text_annotation.text:
        return resp.full_text_annotation.text.strip()
    return "인식된 글자가 없습니다."

# ---------------------------
# 자연스러운 TTS (Google Cloud)
# ---------------------------
def speak(text: str, lang="ko-KR", voice_name="ko-KR-Neural2-B", speaking_rate=0.98, pitch=0.0):
    client = tts.TextToSpeechClient()
    synthesis_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(language_code=lang, name=voice_name)
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16,
        speaking_rate=speaking_rate,
        pitch=pitch,
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice_params, audio_config=audio_config
    )
    with open(AUDIO_FILE_PATH, "wb") as out:
        out.write(response.audio_content)
    try:
        subprocess.run(["aplay", "-q", AUDIO_FILE_PATH], check=True)
    except Exception:
        pass

# ---------------------------
# 메인 루프
# ---------------------------
def main(bus: FrameBus, camera_manager):
    OPEN_EXPIRE_SEC = 2.0
    COOLDOWN_SEC    = 1.2
    last_open_time = 0.0
    last_trigger_time = 0.0
    fist_start_time = None

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    sub_id = bus.subscribe(queue_size=1)
    q = bus.get_queue(sub_id)

    while True:
        try:
            frame_rgb = q.get(timeout=0.5)
        except Exception:
            frame_rgb = None
        if frame_rgb is None:
            continue
        results = hands.process(frame_rgb)
        state_txt = "SHOW OPEN HAND"
        state_color = COUNTDOWN_COLOR
        trigger_now = False
        now = time.time()

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_open_hand(hand_landmarks):
                last_open_time = now
                fist_start_time = None
                state_txt = "READY: OPEN DETECTED"
                state_color = COUNTDOWN_COLOR

            elif is_fist(hand_landmarks):
                if (now - last_open_time) <= OPEN_EXPIRE_SEC and (now - last_trigger_time) >= COOLDOWN_SEC:
                    trigger_now = True
                    last_trigger_time = now
                    fist_start_time = None
                    state_txt = "CAPTURE!"
                    state_color = CAPTURE_COLOR
                else:
                    if fist_start_time is None:
                        fist_start_time = now
                    held = now - fist_start_time
                    remain = max(0.0, FIST_HOLD_TIME - held)
                    if held >= FIST_HOLD_TIME and (now - last_trigger_time) >= COOLDOWN_SEC:
                        trigger_now = True
                        last_trigger_time = now
                        fist_start_time = None
                        state_txt = "CAPTURE!"
                        state_color = CAPTURE_COLOR
                    else:
                        state_txt = f"HOLD FIST {remain:.1f}s"
                        state_color = COUNTDOWN_COLOR
            else:
                fist_start_time = None

        cv2.putText(frame_rgb, state_txt, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
        cv2.imshow(WINDOW_TITLE, frame_rgb)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if trigger_now:
            # camera_manager를 통해 고해상도 캡처 요청
            camera_manager.request_still_capture(IMAGE_FILE_PATH)

            # OCR 및 TTS 로직은 그대로 유지
            text = detect_text(IMAGE_FILE_PATH)
            print(f"[OCR]\n{text}\n")
            try:
                speak(text, lang="ko-KR", voice_name="ko-KR-Neural2-B", speaking_rate=0.98, pitch=0.0)
            except Exception as e:
                print("[TTS ERROR]", e)

    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    from Common.frame_bus import FrameBus
    from Common.capture import CameraManager
    
    bus = FrameBus()
    # 단독 실행 시 CameraManager와 큐를 임시로 생성
    still_capture_queue = queue.Queue()
    camera_manager = CameraManager(bus, still_capture_queue)
    # 별도 스레드에서 run_walk_mode를 실행하여 충돌 방지
    threading.Thread(target=camera_manager.start, daemon=True).start()
    main(bus, camera_manager)
