#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, sys, threading, time, queue
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw

from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

# --- 모드 전용 설정 ---
from WalkingMode.config import (
    SCORE_THRESHOLD, DEDUP_DISTANCE,
    WINDOW_TITLE, SHOW_WINDOW, SHOW_FPS, FONT_COLOR, BOX_COLOR, DISPLAY_SIZE,
    WALKING_MODEL_PATH, WALKING_LABEL_PATH, DANGER_THRESHOLDS, CONF_THRESHOLD
)
from WalkingMode.deep_learning import PersonDetector
from WalkingMode.risk import RiskLevel, RiskClassifier
from WalkingMode.notifier import Notifier
from Common.img_processing import DistanceEstimator, detect_traffic_light_color

# --- 공용 캡처 ---
from Common.frame_bus import FrameBus

# --------------------------------------------------------------------------
# 모델/라벨 로드
# --------------------------------------------------------------------------
script_dir = Path(__file__).parent
labels = read_label_file(str(script_dir / WALKING_LABEL_PATH))
interpreter = make_interpreter(str(script_dir / WALKING_MODEL_PATH))
interpreter.allocate_tensors()

# --------------------------------------------------------------------------
# WalkingApp
# --------------------------------------------------------------------------
class WalkingApp:
    def __init__(self, bus: FrameBus):
        self.bus = bus
        self.sub_id = bus.subscribe(queue_size=1)
        self.running = True
        self.last_frame_time = time.perf_counter()

        self.detector = PersonDetector(conf=CONF_THRESHOLD)
        self.estimator = DistanceEstimator()
        self.classifier = RiskClassifier()
        self.notifier = Notifier()

    def run(self):
        q = self.bus.get_queue(self.sub_id)

        while self.running:
            try:
                frame_rgb = q.get(timeout=0.5)
                if frame_rgb is None:
                    break
            except queue.Empty:
                continue

            # FPS 갱신
            now = time.perf_counter()
            framerate = 1.0 / max(1e-6, (now - self.last_frame_time))
            self.last_frame_time = now

            # OpenCV 색공간 일치 (RGB를 BGR로 변환)
            # PersonDetector가 BGR 입력을 기대하므로 변환 필요
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 객체 감지
            detections = self.detector.detect(frame_bgr)
            
            # 시각화 및 위험 판단은 BGR 프레임에서 수행
            for (x1, y1, x2, y2, cx, cy, class_id, class_name) in detections:
                if class_id == 9:  # traffic light
                    color_name = detect_traffic_light_color(frame_bgr, (x1, y1, x2, y2, class_id, class_name))
                    color = (0, 255, 0) if color_name == "GREEN" else (0, 0, 255)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_bgr, f"{color_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    continue

                dist_mm = self.estimator.estimate_from_bbox((x1, y1, x2, y2, class_id, class_name),
                                                           image_height=frame_bgr.shape[0])
                risk = self.classifier.classify(dist_mm, class_id=class_id)
                self.notifier.notify(risk)

                if risk == RiskLevel.DANGER:
                    color = (0, 0, 255)
                elif risk == RiskLevel.CAUTION:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)
                
                text = f"{class_name} {dist_mm/1000:.2f}m"

                cv2.circle(frame_bgr, (cx, cy), 3, (255, 0, 0), -1)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # FPS 표시
            if SHOW_FPS:
                cv2.putText(frame_bgr, f"{int(framerate):02d} fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 디스플레이
            if SHOW_WINDOW:
                disp = cv2.resize(frame_bgr, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imshow(WINDOW_TITLE, disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
    
    def stop(self):
        self.running = False
        self.bus.unsubscribe(self.sub_id)

# --------------------------------------------------------------------------
# 메인 함수
# --------------------------------------------------------------------------
def main(bus: FrameBus):
    app = WalkingApp(bus)
    app.run()

if __name__ == "__main__":
    from Common.frame_bus import FrameBus
    bus = FrameBus()
    main(bus)
