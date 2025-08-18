#!/usr/bin/python3

import math
import sys
import threading
import time
import libcamera
from pathlib import Path
from PIL import Image, ImageDraw
from picamera2 import Picamera2
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import cv2
import numpy as np

# Configure filesystem
script_path = Path(__file__)
script_dir = script_path.parent

# Camera resolution (no Display HAT)
width, height = 480, 640

# Configure camera
picam2 = Picamera2()
capture_config = picam2.create_still_configuration(
    main={"size": (width, height), "format": "RGB888"},
    lores=None,
    raw=None,
    colour_space=libcamera.ColorSpace.Raw(),
    buffer_count=6,
    queue=True
)
picam2.configure(capture_config)
picam2.start()

# ===== EfficientDet-Lite0(food)로 변경 =====
# Labels: ./model/food_label.txt (라인별 클래스명)
# Model : ./model/food_effi_lite0_int8_edgetpu.tflite
image_buffer = Image.new("RGB", (width, height))
labels = read_label_file(str(script_dir / "./model/labels.txt"))
interpreter = make_interpreter(str(script_dir / "./model/food_effi_lite0_200ep_int8.tflite"))
interpreter.allocate_tensors()

# 추론 및 표시 상태
detected_objs = []
inference_latency = sys.float_info.max

def is_duplicate(center1, center2):
    dist_thresh = 15
    return dist_thresh >= math.dist(center1, center2)

def run_interpreter():
    global image_buffer, detected_objs, inference_latency

    start = time.perf_counter()
    # PIL 이미지 버퍼를 모델 입력 크기에 맞춰 리사이즈(set_resized_input가 scale 반환)
    _, scale = common.set_resized_input(
        interpreter,
        image_buffer.size,
        lambda size: image_buffer.resize(size, Image.Resampling.LANCZOS)
    )
    interpreter.invoke()
    inference_latency = time.perf_counter() - start

    # EfficientDet 출력 파싱 + 좌표 스케일 보정
    # score_threshold는 필요에 맞춰 조절 (예: 0.2~0.5)
    objs = detect.get_objects(interpreter, score_threshold=0.15, image_scale=scale)

    # 같은 클래스에서 너무 가까운 중복 박스 제거(선택)
    dedup_map = {}
    filtered_objs = []

    for obj in objs:
        bbox = obj.bbox
        center = ((bbox.xmax + bbox.xmin) / 2, (bbox.ymax + bbox.ymin) / 2)

        bucket = dedup_map.get(obj.id)
        if bucket is not None:
            if any(is_duplicate(center, other_center) for other_center in bucket):
                continue
        else:
            dedup_map[obj.id] = []

        dedup_map[obj.id].append(center)
        filtered_objs.append((obj, bbox))

    detected_objs = filtered_objs

# 첫 추론 스레드 준비
inference_thread = threading.Thread(target=run_interpreter)

last_frame_time = time.perf_counter()
framerate = 0

# Main loop
try:
    while True:
        # Picamera2 → PIL.Image (RGB888)
        frame_pil = picam2.capture_image()

        # 추론 스레드가 비어 있으면 새 프레임으로 갱신 후 시작
        if not inference_thread.is_alive():
            image_buffer.paste(frame_pil)
            inference_thread = threading.Thread(target=run_interpreter)
            inference_thread.start()

        # 그리기
        draw = ImageDraw.Draw(frame_pil)

        for obj, bbox in detected_objs:
            # 라벨 이름 가져오기 (read_label_file은 인덱스→이름 dict 반환)
            cls_name = labels.get(obj.id, str(obj.id))
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline='yellow', width=2)
            draw.text(
                (bbox.xmin + 10, bbox.ymin + 10),
                f"{cls_name}\n{obj.score:.2f}",
                fill='yellow'
            )

        draw.text(
            (10, 10),
            f"{int(framerate):02d} fps\n{inference_latency*1000:.2f} ms",
            fill='white'
        )

        # PIL → OpenCV (BGR)로 변환하여 화면 표시
        frame_np = np.array(frame_pil)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        cv2.imshow('Food EfficientDet EdgeTPU', frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        this_frame_time = time.perf_counter()
        framerate = 1 / (this_frame_time - last_frame_time)
        last_frame_time = this_frame_time

finally:
    if inference_thread.is_alive():
        inference_thread.join()
    cv2.destroyAllWindows()