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

def detect_traffic_light_color(image, bbox):
    """
    신호등 색상 감지 함수 (ROI → HSV 변환 후 분석)
    입력: BGR 이미지와 bbox(x1, y1, x2, y2, class_id, class_name)
    출력: "RED", "GREEN", "UNKNOWN"
    """
    x1, y1, x2, y2, class_id, *_ = bbox
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 빨간불 HSV 범위 (두 개)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 초록불 HSV 범위 (확장된 범위)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if DEBUG:
        print(f"[신호등 HSV] red={red_pixels}, green={green_pixels}")

    if red_pixels > green_pixels and red_pixels > 100:
        return "RED"
    elif green_pixels > red_pixels and green_pixels > 50:
        return "GREEN"
    else:
        return "UNKNOWN"


