class RiskLevel:
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    DANGER = "DANGER"

class RiskClassifier:
    def __init__(self, caution_threshold_mm=2000, danger_threshold_mm=1000):
        self.caution_threshold = caution_threshold_mm
        self.danger_threshold = danger_threshold_mm

    def classify(self, distance_mm):
        if distance_mm < 0:
            return RiskLevel.SAFE
        if distance_mm < self.danger_threshold:
            return RiskLevel.DANGER
        elif distance_mm < self.caution_threshold:
            return RiskLevel.CAUTION
        else:
            return RiskLevel.SAFE