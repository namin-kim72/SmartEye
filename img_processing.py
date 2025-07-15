import cv2
import numpy as np
import math

# =============================================================================
# 하이퍼파라미터 설정 (안경 삽입형 카메라 기준)
# =============================================================================
CAMERA_HEIGHT = 1700      # 카메라 높이 (mm)
CAMERA_ANGLE = 25         # 하향 각도 (도)
CAMERA_FOV = 60           # 수직 화각 (도)
FRAME_HEIGHT = 480        # 이미지 높이 (픽셀)
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
        bottom_dist = self.height * math.tan(math.radians(bottom_angle))

        slope = (bottom_dist - center_dist) / (bottom_y - center_y)
        intercept = center_dist - slope * center_y

        y_coords = np.linspace(0, self.image_height, self.image_height + 1)
        distances = slope * y_coords + intercept

        return {
            'slope': slope,
            'intercept': intercept,
            'center_y': center_y,
            'distances': distances
        }

    def _apply_correction(self, raw_m):
        return raw_m * 0.88

    def estimate_from_pixel(self, y_pixel):
        y = max(0, min(y_pixel, self.image_height))
        slope = self.mapping['slope']
        intercept = self.mapping['intercept']
        raw_mm = slope * y + intercept
        corrected_m = self._apply_correction(raw_mm / 1000)
        return corrected_m * 1000

    def estimate_from_bbox(self, bbox):
        _, _, _, y2 = bbox
        dist_mm = self.estimate_from_pixel(y2)
        return {
            'distance_mm': dist_mm,
            'distance_m': dist_mm / 1000,
            'is_safe': dist_mm > SAFE_DISTANCE_MM
        }

def enhance_image(image):
    h, w = image.shape[:2]
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32)
    D = np.array([-0.2, 0.08, 0, 0], dtype=np.float32)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    undistorted = cv2.undistort(image, K, D, None, new_K)

    gamma = 2.0
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    bright = cv2.LUT(undistorted, table)

    ycrcb = cv2.cvtColor(bright, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def estimate_distances(bboxes):
    estimator = DistanceEstimator()
    distances = []
    for bbox in bboxes:
        try:
            result = estimator.estimate_from_bbox(bbox)
            dist_m = result['distance_m']
            distances.append(max(0.1, min(dist_m, 20.0)))
        except Exception as e:
            print(f"거리 계산 오류 - bbox: {bbox}, 에러: {e}")
            distances.append(1.0)
    return distances
