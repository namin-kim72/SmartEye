import cv2
from ultralytics import YOLO

class PersonDetector:
    _detector = None  # 싱글톤 인스턴스 저장 변수

    def __new__(cls, *args, **kwargs):
        if cls._detector is None:
            cls._detector = super(PersonDetector, cls).__new__(cls)
        return cls._detector
        
    def __init__(self, model_path="yolov8n.pt", conf=0.3):
        if not hasattr(self, "model"):
            self.model = YOLO(model_path)
            self.conf = conf

    def detect(self, image):
        """
        BGR 이미지를 받아서 신호등(class_id=9)만 탐지하고
        (x1, y1, x2, y2, class_id, class_name) 형태의 바운딩 박스 리스트 반환
        """
        results = self.model(image, conf=self.conf, classes=[9])
        boxes = []

        if results and results[0].boxes is not None:
            for box_obj in results[0].boxes:
                x1, y1, x2, y2 = map(int, box_obj.xyxy[0])
                class_id = int(box_obj.cls)
                class_name = self.model.names[class_id]
                boxes.append((x1, y1, x2, y2, class_id, class_name))
        return boxes
