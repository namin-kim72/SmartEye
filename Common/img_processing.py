# Common/img_processing.py
import cv2
import numpy as np
import math
from Common.config import CAMERA_ROTATE_CCW_90

# =============================================================================
# 거리 추정 기본 파라미터
# =============================================================================
CAMERA_HEIGHT_MM = 1700     # 카메라 높이 (mm)
CAMERA_TILT_DEG = 0         # 카메라 기울기 (도)
CAMERA_VFOV_DEG = 72.4      # 수직 화각 (도)
DEFAULT_FRAME_HEIGHT = 640  # 기본 프레임 높이 (px)
DEBUG = False

class DistanceEstimator:
    """
    픽셀 y좌표 -> 지면까지 거리 추정
    - mm, m 단위 모두 지원
    - bbox 하단(y2) 기준
    """
    def __init__(self, height_mm=CAMERA_HEIGHT_MM, tilt_deg=CAMERA_TILT_DEG,
                 vfov_deg=CAMERA_VFOV_DEG, image_height=DEFAULT_FRAME_HEIGHT, debug=DEBUG):
        self.height_mm    = float(height_mm)
        self.tilt_deg     = float(tilt_deg)
        self.vfov_deg     = float(vfov_deg)
        self.image_height = int(image_height)
        self.debug        = debug
        self.mapping      = self._build_mapping(self.image_height)

    def _build_mapping(self, H: int):
        cy = H / 2.0
        ys = np.linspace(0, H, H + 1)
        alpha = ((ys - cy) / (H / 2.0)) * (self.vfov_deg / 2.0)
        phi = self.tilt_deg + alpha
        phi = np.clip(phi, 0.5, 89.5)
        d_mm = self.height_mm / np.tan(np.deg2rad(phi))
        return {"H": H, "cy": cy, "d_mm": d_mm}

    def _ensure_height(self, image_height: int):
        if image_height != self.mapping["H"]:
            self.mapping = self._build_mapping(image_height)

    def estimate_from_pixel_mm(self, y_pixel: int, *, image_height: int | None = None) -> float:
        if image_height is not None:
            self._ensure_height(image_height)
        H = self.mapping["H"]
        y = int(np.clip(y_pixel, 0, H))
        d_mm = float(self.mapping["d_mm"][y])
        d_mm *= 0.88  # 보정계수
        return d_mm

    def estimate_from_pixel_m(self, y_pixel: int, *, image_height: int | None = None) -> float:
        return self.estimate_from_pixel_mm(y_pixel, image_height=image_height) / 1000.0

    def estimate_from_bbox_mm(self, bbox, *, image_height: int | None = None) -> float:
        _, _, _, y2, *_ = bbox
        return self.estimate_from_pixel_mm(y2, image_height=image_height)

    def estimate_from_bbox_m(self, bbox, *, image_height: int | None = None) -> float:
        return self.estimate_from_bbox_mm(bbox, image_height=image_height) / 1000.0


# =============================================================================
# 영상 전처리
# =============================================================================
def enhance_image(image):
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
    보행자 신호등 색상 감지
    반환: "RED" | "GREEN" | "UNKNOWN"
    """
    x1, y1, x2, y2, *_ = bbox
    h, w = image_bgr.shape[:2]

    # ROI 축소
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
    Hc, Sc, Vc = cv2.split(hsv)

    # 불빛 특성
    s_mask = cv2.inRange(Sc, 100, 255)
    v_mask = cv2.inRange(Vc, 130, 255)
    sv = cv2.bitwise_and(s_mask, v_mask)

    # 색 마스크
    red1 = cv2.inRange(hsv, (0,   80, 80), (12, 255, 255))
    red2 = cv2.inRange(hsv, (165, 80, 80), (180, 255, 255))
    red  = cv2.bitwise_or(red1, red2)
    green = cv2.inRange(hsv, (55, 80, 80), (85, 255, 255))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red   = cv2.morphologyEx(cv2.bitwise_and(red, sv), cv2.MORPH_OPEN, k)
    green = cv2.morphologyEx(cv2.bitwise_and(green, sv), cv2.MORPH_OPEN, k)

    # 상/하 분리
    Hroi = roi.shape[0]
    top = slice(0, Hroi // 2)
    bot = slice(Hroi // 2, Hroi)

    red_top    = int(np.count_nonzero(red[top, :]))
    red_bot    = int(np.count_nonzero(red[bot, :]))
    green_top  = int(np.count_nonzero(green[top, :]))
    green_bot  = int(np.count_nonzero(green[bot, :]))
    red_sum    = red_top + red_bot
    green_sum  = green_top + green_bot

    # 임계값
    area = max(1, roi.shape[0] * roi.shape[1])
    PIX_MIN_R = max(40, int(area * 0.002))
    PIX_MIN_G = max(35, int(area * 0.0015))

    if debug:
        print(f"[DBG] red_top={red_top}, green_bot={green_bot}, thrR={PIX_MIN_R}, thrG={PIX_MIN_G}")

    # 판정
    if green_bot > PIX_MIN_G and green_bot > red_top * 0.7 and green_sum > red_sum * 0.5:
        return "GREEN"
    if red_top > PIX_MIN_R and red_top > green_bot * 0.7 and red_sum > green_sum * 0.5:
        return "RED"

    return "UNKNOWN"
