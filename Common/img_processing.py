# Common/img_processing.py
import cv2
import numpy as np
import math
from Common.config import CAMERA_ROTATE_CCW_90

# =============================================================================
# 거리 추정 (카메라 매개변수는 필요 시 수정)
# =============================================================================
CAMERA_HEIGHT_MM = 1700     # 카메라 높이(mm)
CAMERA_TILT_DEG = 0         # 하향 각도(도)
CAMERA_VFOV_DEG = 72.4      # 수직 화각(도)
DEFAULT_FRAME_HEIGHT = 640  # 기본 프레임 높이(px)
DEBUG = False

class DistanceEstimator:
    def __init__(self, height_mm=CAMERA_HEIGHT_MM, tilt_deg=CAMERA_TILT_DEG,
                 vfov_deg=CAMERA_VFOV_DEG, image_height=DEFAULT_FRAME_HEIGHT, debug=DEBUG):
        self.height = float(height_mm)              # mm
        self.tilt = float(tilt_deg)                 # deg (수평선 기준 아래 +)
        self.vfov = float(vfov_deg)                 # deg (수직 FOV)
        self.image_height = int(image_height)       # px
        self.debug = debug
        self.mapping = self._build_mapping()

    def _build_mapping(self):
        """
        각도 선형 + tan 모델:
          φ(y) = tilt + ((y - H/2)/H) * vfov * 2    (deg)
          d_raw(y) = height / tan(φ(y))
        """
        H = self.height
        img_h = float(self.image_height)

        # y=0..img_h
        y_coords = np.arange(0, self.image_height + 1, dtype=np.float32)
        r = (y_coords - img_h/2.0) / img_h            # [-0.5, +0.5]
        phi = self.tilt + (r * self.vfov) * 2.0       # deg

        # 수치 안전장치 (0°/90° 근접 폭발 방지)
        phi = np.clip(phi, 0.01, 89.9)
        distances_raw_mm = H / np.tan(np.deg2rad(phi))  # mm

        if self.debug:
            phi_top = self.tilt - self.vfov/2.0
            phi_bot = self.tilt + self.vfov/2.0
            print(f"[DBG] φ_top={phi_top:.2f}°, φ_center={self.tilt:.2f}°, φ_bottom={phi_bot:.2f}°")

        return {
            "y_coords": y_coords,
            "phi_deg": phi,
            "dist_raw_mm": distances_raw_mm,  # 보정 전
        }

    def _apply_correction(self, raw_m: float) -> float:
        """
        경험적 보정.
        - 간단: 고정 배수(≈0.86 권장)
        - 정밀: 표본으로 선형(a·raw + b) 피팅
        """
        USE_LINEAR = False
        if not USE_LINEAR:
            CORRECTION_FACTOR = 0.86
            return raw_m * CORRECTION_FACTOR
        # 예시: 표본 2.33→2.0, 3.67→3.0, 4.49→4.0 로 피팅한 값
        a, b = 0.9083772569249747, -0.17629247504766127
        return max(0.0, a * raw_m + b)

    def estimate_from_pixel(self, y_pixel):
        # 개별 y에 대해 즉시 계산 (매핑 테이블 없이도 동일 로직)
        y = float(np.clip(y_pixel, 0, self.image_height))
        H = self.height
        img_h = float(self.image_height)
        r = (y - img_h/2.0) / img_h
        phi = self.tilt + (r * self.vfov) * 2.0
        phi = float(np.clip(phi, 0.01, 89.9))

        raw_m = (H / math.tan(math.radians(phi))) / 1000.0
        corrected_m = self._apply_correction(raw_m)
        return corrected_m  # m 단위

    def estimate_from_bbox(self, bbox, *, image_height=None):
        if image_height and image_height != self.image_height:
            tmp = DistanceEstimator(
                height_mm=self.height,
                tilt_deg=self.tilt,
                vfov_deg=self.vfov,
                image_height=image_height,
                debug=self.debug,
            )
            return tmp.estimate_from_bbox(bbox)
        _, _, _, y2, *_ = bbox
        return self.estimate_from_pixel(y2)


# =============================================================================
# 영상 전처리 유틸
# =============================================================================
def enhance_image(image):
    """
    가벼운 공용 전처리:
    - 필요 시 CCW 90° 회전(공용 설정 기반)
    - 추가 밝기/대비/히스토그램 평활화는 호출처에서 선택
    """
    out = image
    if CAMERA_ROTATE_CCW_90:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return out

def adaptive_gamma(image, gamma=2.0):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def equalize_luma(image_bgr):
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

# =============================================================================
# 신호등 색상 감지
# =============================================================================
def detect_traffic_light_color(image_bgr, bbox, debug=False):
    """
    신호등 색상 감지(보행자 신호 특화)
    bbox = (x1, y1, x2, y2, class_id, ...)
    반환: "RED" | "GREEN" | "UNKNOWN"
    """
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



