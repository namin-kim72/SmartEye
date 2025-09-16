# WalkingMode/walk_processing.py

import cv2
import numpy as np
import math
from WalkingMode.config import CAMERA_ROTATE_CCW_90

# =============================================================================
# 거리 추정기 (각도 센서 기반으로 수정)
# =============================================================================
# 하이퍼 파라미터 (보행 모드에 맞게 수정)
CAMERA_HEIGHT_MM = 1700   # 카메라 높이 (mm)
CAMERA_VFOV = 72.4        # 카메라 수직 화각 (도)
DEFAULT_FRAME_HEIGHT = 640 # 이미지 높이 (픽셀)
DEFAULT_TILT_ANGLE = 25   # 기본 카메라 설치 각도 (도)

class DistanceEstimator:
    def __init__(self, height_mm=CAMERA_HEIGHT_MM, vfov_deg=CAMERA_VFOV, image_height=DEFAULT_FRAME_HEIGHT, debug=False):
        self.height = height_mm
        self.vfov = vfov_deg
        self.image_height = image_height
        self.debug = debug
    
    def _calculate_distance_correction(self, raw_distance_m):
        CORRECTION_FACTOR = 0.88
        return raw_distance_m * CORRECTION_FACTOR

    def estimate_from_bbox(self, bbox, *, image_height=None, tilt_deg=None) -> float:
        """
        바운딩 박스의 하단 Y좌표와 실시간 기울기(tilt_deg)를 사용하여 거리 추정 (mm 단위).
        tilt_deg가 None이면 기본 설치 각도만 사용.
        """
        if image_height and image_height != self.image_height:
            tmp = DistanceEstimator(
                height_mm=self.height, vfov_deg=self.vfov, image_height=image_height, debug=self.debug,
            )
            return tmp.estimate_from_bbox(bbox, tilt_deg=tilt_deg)

        _, _, _, y2, *_ = bbox
        foot_y = y2

        img_h = float(self.image_height)
        r = (foot_y - img_h/2.0) / img_h

        # pitch 센서 값이 있으면 설치 오프셋(25°)을 더해 사용
        base_tilt = DEFAULT_TILT_ANGLE
        effective_tilt = base_tilt + (tilt_deg if tilt_deg is not None else 0.0)

        phi = effective_tilt + (r * self.vfov)
        phi = float(np.clip(phi, 0.01, 89.9))

        raw_distance_mm = self.height / math.tan(math.radians(phi))
        raw_distance_m = raw_distance_mm / 1000.0

        corrected_distance_m = self._calculate_distance_correction(raw_distance_m)
        corrected_distance_mm = corrected_distance_m * 1000

        corrected_distance_mm = max(100.0, min(corrected_distance_mm, 20000.0))
        
        return float(corrected_distance_mm)


# =============================================================================
# 신호등 색상 감지
# =============================================================================
def detect_traffic_light_color(image_bgr, bbox, debug=False):
    x1, y1, x2, y2, *_ = bbox
    h, w = image_bgr.shape[:2]

    m = 0.12
    x1 = max(0, int(x1 + (x2 - x1) * m))
    x2 = min(w - 1, int(x2 - (x2 - x1) * m))
    y1 = max(0, int(y1 + (y2 - y1) * m))
    y2 = min(h - 1, int(y2 - (y2 - y1) * m))

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return "UNKNOWN"

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    s_mask = cv2.inRange(S, 100, 255)
    v_mask = cv2.inRange(V, 130, 255)
    sv = cv2.bitwise_and(s_mask, v_mask)

    red1 = cv2.inRange(hsv, (0, 80, 80), (12, 255, 255))
    red2 = cv2.inRange(hsv, (165, 80, 80), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)
    green = cv2.inRange(hsv, (55, 80, 80), (85, 255, 255))

    red = cv2.bitwise_and(red, sv)
    green = cv2.bitwise_and(green, sv)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)

    Hroi = roi.shape[0]
    top = slice(0, Hroi // 2)
    bot = slice(Hroi // 2, Hroi)

    red_top = int(np.count_nonzero(red[top, :]))
    red_bot = int(np.count_nonzero(red[bot, :]))
    green_top = int(np.count_nonzero(green[top, :]))
    green_bot = int(np.count_nonzero(green[bot, :]))
    red_sum = red_top + red_bot
    green_sum = green_top + green_bot

    area = roi.shape[0] * roi.shape[1]
    PIX_MIN = max(40, int(area * 0.002))
    PIX_MIN_G = max(35, int(area * 0.0015))

    if debug:
        print(f"[DBG] red_top={red_top}, red_bot={red_bot}, green_top={green_top}, "
              f"green_bot={green_bot}, red_sum={red_sum}, green_sum={green_sum}, "
              f"thrR={PIX_MIN}, thrG={PIX_MIN_G}")

    if green_bot > PIX_MIN_G and green_bot > red_top * 0.7 and green_sum > red_sum * 0.5:
        return "GREEN"
    if (red_top > PIX_MIN) or (red_bot > PIX_MIN and green_sum < PIX_MIN_G):
        return "RED"
    if red_sum > green_sum * 1.8 and red_sum > PIX_MIN:
        return "RED"
    if green_sum > red_sum * 1.5 and green_sum > PIX_MIN_G:
        return "GREEN"
    return "UNKNOWN"
