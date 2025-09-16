#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, sys, threading, time, queue
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw

from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from WalkingMode.config import (
    SCORE_THRESHOLD, DEDUP_DISTANCE, WINDOW_TITLE, SHOW_WINDOW, SHOW_FPS,
    FONT_COLOR, BOX_COLOR, DISPLAY_SIZE, WALKING_MODEL_PATH, WALKING_LABEL_PATH,
    DANGER_THRESHOLDS, CONF_THRESHOLD, USE_ANGLE_SENSOR
)
from WalkingMode.deep_learning import PersonDetector, PitchSensor
from WalkingMode.risk import RiskLevel, RiskClassifier
from WalkingMode.notifier import Notifier
from WalkingMode.walk_processing import DistanceEstimator, detect_traffic_light_color
from Common.frame_bus import FrameBus
from Common.utils import graceful_exit, speak
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
    def __init__(self, bus: FrameBus, button_mgr):
        self.bus = bus
        self.sub_id = bus.subscribe(queue_size=1)
        self.running = True
        self.last_frame_time = time.perf_counter()
        
        self.detector = PersonDetector(conf=CONF_THRESHOLD)
        self.estimator = DistanceEstimator()
        self.classifier = RiskClassifier()
        self.notifier = Notifier()
        self.pitch_sensor = PitchSensor() if USE_ANGLE_SENSOR else None
        self.button_mgr = button_mgr
        
    def run(self):
        q = self.bus.get_queue(self.sub_id)
        
        # 각도 센서 스레드 시작
        if self.pitch_sensor:
            self.pitch_sensor.start()

        while self.running:
            # ---------------------------
            # 버튼 이벤트 확인
            # ---------------------------
            event = self.button_mgr.get_event()
            if event == "LONG_PRESS":
                cv2.destroyAllWindows()
                self.bus.unsubscribe(self.sub_id)
                graceful_exit()
            elif event == "SHORT_PRESS":
                cv2.destroyAllWindows()
                self.bus.unsubscribe(self.sub_id)
                return  # 메인으로 복귀 → 다음 모드 실행
                
            try:
                frame_bgr = q.get(timeout=0.5)
                if frame_bgr is None:
                    break
            except queue.Empty:
                continue

            now = time.perf_counter()
            framerate = 1.0 / max(1e-6, (now - self.last_frame_time))
            self.last_frame_time = now

            detections = self.detector.detect(frame_bgr)
            current_tilt_angle = self.pitch_sensor.pitch_deg if self.pitch_sensor else None

            if current_tilt_angle is not None:
                print(f"Pitch Sensor Tilt: {current_tilt_angle:.2f} degrees")
            else:
                print("Pitch Sensor: No data")

            for detection in detections:
                x1, y1, x2, y2, cx, cy, class_id, class_name = detection
                
                if class_id == 9:
                    color_name = detect_traffic_light_color(frame_bgr, detection)
                    color = (0, 255, 0) if color_name == "GREEN" else (0, 0, 255)
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_bgr, f"{color_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    continue

                dist_mm = self.estimator.estimate_from_bbox(
                    detection, 
                    image_height=frame_bgr.shape[0], 
                    tilt_deg=current_tilt_angle
                )
                
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

            if SHOW_FPS:
                cv2.putText(frame_bgr, f"{int(framerate):02d} fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if SHOW_WINDOW:
                disp = cv2.resize(frame_bgr, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imshow(WINDOW_TITLE, disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
    
    def stop(self):
        self.running = False
        self.bus.unsubscribe(self.sub_id)
        if self.pitch_sensor:
            self.pitch_sensor.stop()

def main(bus: FrameBus, button_mgr):
    app = WalkingApp(bus, button_mgr)
    app.run()

if __name__ == "__main__":
    from Common.frame_bus import FrameBus
    bus = FrameBus()
    main(bus)
