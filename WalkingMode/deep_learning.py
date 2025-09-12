# WalkingMode/deep_learning.py

import serial
import time
import cv2
from PIL import Image
from pathlib import Path

from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file

from WalkingMode.config import (
    WALKING_MODEL_PATH, WALKING_LABEL_PATH, TARGET_CLASS_IDS, CONF_THRESHOLD,
    USE_LASER_SENSOR, LASER_SENSOR_PORT, LASER_SENSOR_BAUDRATE, LASER_MAX_DISTANCE_MM
)

class PersonDetector:
    def __init__(self, model_path=None, conf=None):
        script_dir = Path(__file__).parent
        model_path = script_dir / (model_path or WALKING_MODEL_PATH)
        label_path = script_dir / WALKING_LABEL_PATH

        self.interpreter = make_interpreter(str(model_path))
        self.interpreter.allocate_tensors()
        self.input_size = common.input_size(self.interpreter)
        self.threshold = CONF_THRESHOLD if conf is None else conf
        self.target_ids = TARGET_CLASS_IDS
        self.labels = read_label_file(str(label_path))

    def detect(self, image_bgr):
        # 내부에서 RGB 변환 → 리사이즈 → 추론
        orig_h, orig_w = image_bgr.shape[:2]
        image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_resized = image_pil.resize(self.input_size, Image.Resampling.LANCZOS)
        common.set_input(self.interpreter, image_resized)
        self.interpreter.invoke()

        objs = detect.get_objects(self.interpreter, self.threshold)
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]

        boxes = []
        for obj in objs:
            if obj.id not in self.target_ids:
                continue
            bbox = obj.bbox
            x1 = int(bbox.xmin * scale_x); y1 = int(bbox.ymin * scale_y)
            x2 = int(bbox.xmax * scale_x); y2 = int(bbox.ymax * scale_y)
            cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
            boxes.append((x1, y1, x2, y2, cx, cy, obj.id, self.labels.get(obj.id, str(obj.id))))
        return boxes

class LaserDistanceSensor:
    def __init__(self, port=LASER_SENSOR_PORT, baudrate=LASER_SENSOR_BAUDRATE):
        self.ser = None
        if USE_LASER_SENSOR:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=1)
                time.sleep(2)
                print(f"[INFO] 레이저 센서 연결 성공: {port}")
            except serial.SerialException as e:
                print(f"[ERROR] 레이저 센서 연결 실패: {e}")
                self.ser = None

    def read_distance(self):
        if self.ser and self.ser.in_waiting:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line.isdigit():
                    distance_mm = int(line)
                    if 0 < distance_mm <= LASER_MAX_DISTANCE_MM:
                        return distance_mm
            except (UnicodeDecodeError, ValueError) as e:
                print(f"[ERROR] 레이저 센서 데이터 파싱 오류: {e}")
        return None

    def close(self):
        if self.ser:
            self.ser.close()
            print("[INFO] 레이저 센서 연결 종료")
