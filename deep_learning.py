import torch
import cv2

class PersonDetector:
    def __init__(self, model_path='yolov5n.pt', conf=0.3):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = conf  # confidence threshold
        self.model.classes = [0]  # class 0 = person only

    def detect(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, size=640)

        boxes = []
        for *xyxy, conf, cls in results.xyxy[0]:  # torch.Tensor
            x1, y1, x2, y2 = map(int, xyxy)
            boxes.append((x1, y1, x2, y2))
        return boxes
