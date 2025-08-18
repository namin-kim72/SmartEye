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
    """
    픽셀 y좌표(특히 bbox 하단 y) -> 지면까지의 대략적 거리(m) 추정기.
    안전/위험 판단은 공용이 아닌 '보행모드 risk 모듈'에서 한다.
    """
    def __init__(self, height_mm=CAMERA_HEIGHT_MM, tilt_deg=CAMERA_TILT_DEG,
                 vfov_deg=CAMERA_VFOV_DEG, image_height=DEFAULT_FRAME_HEIGHT, debug=DEBUG):
        self.height = height_mm
        self.tilt = tilt_deg
        self.vfov = vfov_deg
        self.image_height = image_height
        self.debug = debug
        self.mapping = self._build_mapping()

    def _build_mapping(self):
        center_y = self.image_height // 2
        center_angle = 90 - self.tilt
        center_dist = self.height * math.tan(math.radians(center_angle))

        bottom_y = self.image_height
        bottom_angle = center_angle - (self.vfov / 2)
        if bottom_angle <= 1:
            bottom_angle = 1
        bottom_dist = self.height * math.tan(math.radians(bottom_angle))

        slope = (bottom_dist - center_dist) / max(1, (bottom_y - center_y))
        intercept = center_dist - slope * center_y

        y_coords = np.linspace(0, self.image_height, self.image_height + 1)
        distances = slope * y_coords + intercept

        return {
            "slope": slope,
            "intercept": intercept,
            "center_y": center_y,
            "distances": distances,
        }

    def _apply_correction(self, raw_mm):
        # 보정계수(경험적): 피팅 후 필요 시 조정
        return raw_mm * 0.88

    def estimate_from_pixel(self, y_pixel):
        y = max(0, min(int(y_pixel), self.image_height))
        raw_mm = self.mapping["slope"] * y + self.mapping["intercept"]
        corrected_mm = self._apply_correction(raw_mm)
        return corrected_mm / 1000.0   # ✅ 결과를 m 단위로 반환

    def estimate_from_bbox(self, bbox, *, image_height=None):
        """
        bbox = (x1, y1, x2, y2, class_id, ...)
        공용 단계에서는 거리(m)만 반환하고, 위험판단은 호출측(보행모드 risk)이 수행.
        """
        if image_height and image_height != self.image_height:
            # 프레임 높이가 다르면 임시 인스턴스 생성해 사용
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
