# WalkingMode/risk.py
from WalkingMode.config import DANGER_THRESHOLDS

class RiskLevel:
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    DANGER = "DANGER"

class RiskClassifier:
    """
    per-class danger 임계(mm)를 사용.
    caution은 danger의 배수로 산정(기본 1.6배).
    """
    def __init__(self, caution_factor=1.6):
        self.caution_factor = caution_factor

    def _danger_thr(self, class_id):
        return DANGER_THRESHOLDS.get(class_id, DANGER_THRESHOLDS.get("default", 1200))

    def classify(self, distance_mm, class_id=None):
        if distance_mm < 0:
            return RiskLevel.SAFE
        d_thr = self._danger_thr(class_id)
        c_thr = int(d_thr * self.caution_factor)
        if distance_mm < d_thr:
            return RiskLevel.DANGER
        elif distance_mm < c_thr:
            return RiskLevel.CAUTION
        else:
            return RiskLevel.SAFE
