import cv2
from PIL import Image
from config import MODEL_PATH, TARGET_CLASS_IDS, LABEL_PATH
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file

class PersonDetector:
    def __init__(self, model_path=MODEL_PATH, conf=0.4):
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_size = common.input_size(self.interpreter)
        self.threshold = conf
        self.target_ids = TARGET_CLASS_IDS
        self.labels = read_label_file(LABEL_PATH)

    def detect(self, image):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_resized = image_pil.resize(self.input_size, Image.ANTIALIAS)
        common.set_input(self.interpreter, image_resized)
    
        self.interpreter.invoke()
        objs = detect.get_objects(self.interpreter, self.threshold)
    
        boxes = []
        for obj in objs:
            if obj.id not in self.target_ids:
                continue
            bbox = obj.bbox
            boxes.append((
                int(bbox.xmin),
                int(bbox.ymin),
                int(bbox.xmax),
                int(bbox.ymax),
                int(obj.id),
                self.labels.get(obj.id, str(obj.id))
            ))
    
        return boxes
