# WalkingMode/deep_learning.py

import smbus2
import time
import cv2
from PIL import Image
from pathlib import Path
import threading
import math
import sys

from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file

from WalkingMode.config import (
    WALKING_MODEL_PATH, WALKING_LABEL_PATH, TARGET_CLASS_IDS, CONF_THRESHOLD,
    USE_ANGLE_SENSOR
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


class PitchSensor:
    I2C_BUS = 1
    ADDR_PRIMARY = 0x68
    ADDR_ALT     = 0x69
    PWR_MGMT_1   = 0x6B
    SMPLRT_DIV   = 0x19
    CONFIG       = 0x1A
    GYRO_CONFIG  = 0x1B
    ACCEL_CONFIG = 0x1C
    WHO_AM_I     = 0x75
    ACCEL_XOUT_H = 0x3B
    
    ACCEL_SF = 16384.0
    GYRO_SF  = 131.0
    ALPHA = 0.98

    def __init__(self, debug=False):
        self.pitch_deg = 0.0
        self._stop = False
        self._dbg = debug
        self._addr = self.ADDR_PRIMARY
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self.bus = None

        if USE_ANGLE_SENSOR:
            try:
                self.bus = smbus2.SMBus(self.I2C_BUS)
                time.sleep(1)
                print(f"[INFO] 각도 센서 연결 성공: I2C 버스 {self.I2C_BUS}")
            except FileNotFoundError as e:
                print(f"[ERROR] I2C 버스 연결 실패: {e}")
                self.bus = None
    
    def start(self):
        if self.bus and not self._thr.is_alive():
            self._thr.start()

    def stop(self):
        self._stop = True
        if self._thr.is_alive():
            self._thr.join(timeout=1.0)
        
    @staticmethod
    def _s16(h, l):
        v = (h << 8) | l
        return v - 65536 if v > 32767 else v
        
    def _read_block(self, reg, n, retries=5):
        for _ in range(retries):
            try: return self.bus.read_i2c_block_data(self._addr, reg, n)
            except OSError: time.sleep(0.005)
        return None

    def _try_addr(self):
        for addr in (self.ADDR_PRIMARY, self.ADDR_ALT):
            try:
                if self.bus.read_byte_data(addr, self.WHO_AM_I) == 0x68:
                    self._addr = addr
                    return True
            except OSError: pass
        return False
        
    def _loop(self):
        time.sleep(0.3)
        try:
            if not self._try_addr():
                if self._dbg: print("[MPU] device not found; pitch stays 0")
                self.pitch_deg = 0.0
                return
            
            def w(reg, val): self.bus.write_byte_data(self._addr, reg, val)
            w(self.PWR_MGMT_1, 0x80); time.sleep(0.1)
            w(self.PWR_MGMT_1, 0x01); time.sleep(0.1)
            w(self.SMPLRT_DIV, 0x07)
            w(self.CONFIG, 0x03)
            w(self.GYRO_CONFIG, 0x00)
            w(self.ACCEL_CONFIG, 0x00)
            time.sleep(0.1)
            
            gy_bias, valid = 0.0, 0
            for _ in range(150):
                try:
                    b = self._read_block(self.ACCEL_XOUT_H, 14)
                    gy_bias += self._s16(b[10], b[11]) / self.GYRO_SF
                    valid += 1
                    time.sleep(0.003)
                except OSError: continue
            gy_bias = (gy_bias/valid) if valid else 0.0
            
            pitch = 0.0
            last = time.time()
            next_pub_interval = 1.0
            next_pub = time.time() + next_pub_interval
            
            while not self._stop:
                try:
                    b = self._read_block(self.ACCEL_XOUT_H, 14)
                except OSError:
                    time.sleep(0.01)
                    continue
                    
                ax = self._s16(b[0], b[1]) / self.ACCEL_SF
                ay = self._s16(b[2], b[3]) / self.ACCEL_SF
                az = self._s16(b[4], b[5]) / self.ACCEL_SF
                gy = self._s16(b[10], b[11]) / self.GYRO_SF - gy_bias
                
                now = time.time()
                dt = max(1e-3, now - last); last = now
                
                pitch_acc = math.degrees(math.atan2(ay, math.sqrt(ax*ax + az*az)))
                pitch = self.ALPHA*(pitch + gy*dt) + (1-self.ALPHA)*pitch_acc
                
                if now >= next_pub:
                    self.pitch_deg = float(pitch)
                    next_pub += next_pub_interval
                
                time.sleep(0.01)

        except Exception as e:
            if self._dbg: print(f"[MPU] Error in loop: {e}")
            self.pitch_deg = 0.0
