import cv2
import numpy as np
import math
from config import DANGER_THRESHOLDS

# =============================================================================
# 하이퍼파라미터 설정 (안경 삽입형 카메라 기준)
# =============================================================================
CAMERA_HEIGHT = 1700      # 카메라 높이 (mm)
CAMERA_ANGLE = 0         # 하향 각도 (도)
CAMERA_FOV = 72.4           # 수직 화각 (도)
FRAME_HEIGHT = 640       # 이미지 높이 (픽셀)
SAFE_DISTANCE_MM = 2000   # 안전 거리 (mm)
DEBUG = False             # 디버깅 출력 여부

class DistanceEstimator:
    def __init__(self, height=CAMERA_HEIGHT, angle=CAMERA_ANGLE,
                 fov=CAMERA_FOV, image_height=FRAME_HEIGHT, debug=DEBUG):
        self.height = height
        self.angle = angle
        self.fov = fov
        self.image_height = image_height
        self.debug = debug
        self.mapping = self._build_mapping()

    def _build_mapping(self):
        center_y = self.image_height // 2
        center_angle = 90 - self.angle
        center_dist = self.height * math.tan(math.radians(center_angle))

        bottom_y = self.image_height
        bottom_angle = center_angle - (self.fov / 2)
        if bottom_angle <= 1 :
            bottom_angle = 1
        bottom_dist = self.height * math.tan(math.radians(bottom_angle))
        try :
            slope = (bottom_dist - center_dist) / (bottom_y - center_y)
        except ZeroDivisionError:
            slope = 0.001
        intercept = center_dist - slope * center_y

        y_coords = np.linspace(0, self.image_height, self.image_height + 1)
        distances = slope * y_coords + intercept

        return {
            'slope': slope,
            'intercept': intercept,
            'center_y': center_y,
            'distances': distances
        }

    def _apply_correction(self, raw_mm):
        return raw_mm * 0.88

    def estimate_from_pixel(self, y_pixel):
        y = max(0, min(y_pixel, self.image_height))
        slope = self.mapping['slope']
        intercept = self.mapping['intercept']
        raw_mm = slope * y + intercept
        corrected_mm = self._apply_correction(raw_mm)
        return corrected_mm

    def estimate_from_bbox(self, bbox):
        _, _, _, y2, class_id, *_ = bbox
        dist_mm = self.estimate_from_pixel(y2)

        threshold = DANGER_THRESHOLDS.get(class_id, DANGER_THRESHOLDS['default'])
        return {
            'distance_mm': dist_mm,
            'distance_m': dist_mm / 1000,
            'is_safe': dist_mm > threshold
        }


def enhance_image(image):
    undistorted = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    """
    gamma = 2.0
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    bright = cv2.LUT(undistorted, table)

    ycrcb = cv2.cvtColor(bright, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    """
    return undistorted

def estimate_distances(bboxes):
    estimator = DistanceEstimator()
    distances = []
    for bbox in bboxes[:4]:
        try:
            result = estimator.estimate_from_bbox(bbox)
            dist_m = result['distance_m']
            distances.append(max(0.1, min(dist_m, 20.0)))
        except Exception as e:
            print(f"거리 계산 오류 - bbox: {bbox}, 에러: {e}")
            distances.append(1.0)
    return distances

# === 신호등 색상 감지 함수 ===
def detect_traffic_light_color(image, bbox, debug=False):
    import cv2, numpy as np

    x1, y1, x2, y2, class_id, *_ = bbox
    h, w = image.shape[:2]

    # ROI 살짝 축소(테두리/배경 제거)
    m = 0.12
    x1 = max(0, int(x1 + (x2-x1)*m))
    x2 = min(w-1, int(x2 - (x2-x1)*m))
    y1 = max(0, int(y1 + (y2-y1)*m))
    y2 = min(h-1, int(y2 - (y2-y1)*m))

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return "UNKNOWN"

    # 부드럽게 + HSV
    roi_blur = cv2.GaussianBlur(roi, (5,5), 0)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 불빛만 통과(밝고 채도 높은 픽셀)
    s_mask = cv2.inRange(S, 100, 255)
    v_mask = cv2.inRange(V, 130, 255)
    sv = cv2.bitwise_and(s_mask, v_mask)

    # 색 마스크 (조금 보수적으로)
    red1 = cv2.inRange(hsv, (0, 80, 80),   (12, 255, 255))
    red2 = cv2.inRange(hsv, (165,80, 80),  (180,255,255))
    red  = cv2.bitwise_or(red1, red2)

    green = cv2.inRange(hsv, (55, 80, 80), (85, 255,255))

    # 불빛 마스크와 AND
    red   = cv2.bitwise_and(red, sv)
    green = cv2.bitwise_and(green, sv)

    # 노이즈 정리
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    red   = cv2.morphologyEx(red,   cv2.MORPH_OPEN, k)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)

    # 상/하 분리
    Hroi = roi.shape[0]
    top  = slice(0, Hroi//2)
    bot  = slice(Hroi//2, Hroi)

    red_top    = int(np.count_nonzero(red[top, :]))
    red_bot    = int(np.count_nonzero(red[bot, :]))
    green_top  = int(np.count_nonzero(green[top, :]))
    green_bot  = int(np.count_nonzero(green[bot, :]))
    red_sum    = red_top + red_bot
    green_sum  = green_top + green_bot

    # ROI 크기에 따른 적응형 임계값
    area = roi.shape[0] * roi.shape[1]
    PIX_MIN  = max(40, int(area * 0.002))   # 아주 작은 신호등 대비
    PIX_MIN_G = max(35, int(area * 0.0015))

    if debug:
        print(f"[DBG] red_top={red_top}, red_bot={red_bot}, green_top={green_top}, green_bot={green_bot}, "
              f"red_sum={red_sum}, green_sum={green_sum}, thrR={PIX_MIN}, thrG={PIX_MIN_G}")

    # ---------- 판정 규칙(보행 신호 특화) ----------
    # 1) 초록(하단) 우선 규칙
    if green_bot > PIX_MIN_G and green_bot > red_top * 0.7 and green_sum > red_sum * 0.5:
        return "GREEN"

    # 2) 빨강: 상단이 충분히 강하거나(보행자 빨강 사람),
    #    하단에서도 빨강(카운트 숫자)이 많고 초록이 거의 없을 때도 빨강 처리
    if (red_top > PIX_MIN) or (red_bot > PIX_MIN and green_sum < PIX_MIN_G):
        return "RED"

    # 3) 전체 빨강이 초록보다 확실히 우세
    if red_sum > green_sum * 1.8 and red_sum > PIX_MIN:
        return "RED"

    # 4) 전체 초록이 우세
    if green_sum > red_sum * 1.5 and green_sum > PIX_MIN_G:
        return "GREEN"

    return "UNKNOWN"




