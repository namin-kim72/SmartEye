#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
식사모드 (EfficientDet-Lite0 + Coral EdgeTPU + Picamera2)
- 바운딩박스 '하단 중앙' 좌표로 방향 판정
- 구역: 위쪽 3분할(좌전/전방/우전), 아래쪽 2분할(좌/우)
- g: 가이드라인 토글, q: 종료
"""

import math
import sys
import threading
import time
from pathlib import Path

import libcamera
import cv2
import numpy as np
from PIL import Image, ImageDraw
from picamera2 import Picamera2
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


# ============== 기본 설정 ==============
script_path = Path(__file__)
script_dir = script_path.parent

# 카메라 해상도(세로형)
WIDTH, HEIGHT = 480, 640

# 탐지/중복 제거 파라미터
SCORE_THRESHOLD = 0.15
DEDUP_CENTER_DIST = 15  # 같은 클래스 중심 간 거리가 이 값 이하이면 중복으로 간주

# ---- 구역 파라미터(그림과 동일) ----
H_SPLIT   = 0.50      # 수평 분할 위치(화면 높이 비율)
TOP_L     = 1.0/3.0   # 윗부분 좌/전 경계(가로 비율)
TOP_R     = 2.0/3.0   # 윗부분 전/우 경계(가로 비율)
BOTTOM_C  = 0.50      # 아랫부분 좌/우 경계(가로 비율)

# ============== 카메라 설정 ==============
picam2 = Picamera2()
capture_config = picam2.create_still_configuration(
    main={"size": (WIDTH, HEIGHT), "format": "RGB888"},
    lores=None,
    raw=None,
    colour_space=libcamera.ColorSpace.Raw(),
    buffer_count=6,
    queue=True
)
picam2.configure(capture_config)
picam2.start()

# ============== 모델/라벨 ==============
image_buffer = Image.new("RGB", (WIDTH, HEIGHT))
labels = read_label_file(str(script_dir / "./model/labels.txt"))
interpreter = make_interpreter(str(script_dir / "./model/food_effi_lite0_200ep_int8.tflite"))
interpreter.allocate_tensors()

# ============== 전역 상태 ==============
detected_objs = []                 # [(obj, bbox), ...]
inference_latency = sys.float_info.max
last_frame_time = time.perf_counter()
framerate = 0.0
show_guides = False                # 가이드라인 표시 토글


# ============== 유틸 ==============
def is_duplicate(center1, center2, dist_thresh=DEDUP_CENTER_DIST) -> bool:
    return dist_thresh >= math.dist(center1, center2)

def _zone_boundaries(w:int, h:int):
    """그림처럼 경계 좌표 계산 (정수 픽셀 반환)"""
    y_h = int(h * H_SPLIT)
    x_top_l = int(w * TOP_L)
    x_top_r = int(w * TOP_R)
    x_bot_c = int(w * BOTTOM_C)
    return x_top_l, x_top_r, x_bot_c, y_h

def direction_from_bottom_center(x_bc: float, y_bc: float,
                                 frame_w: int, frame_h: int) -> str:
    """
    방향 규칙:
      - y <= y_h(윗부분): x < x_top_l → 좌측전방, x_top_l~x_top_r → 전방, x > x_top_r → 우측전방
      - y >  y_h(아랫부분): x < x_bot_c → 좌측, x >= x_bot_c → 우측
    """
    x_top_l, x_top_r, x_bot_c, y_h = _zone_boundaries(frame_w, frame_h)
    if y_bc <= y_h:
        if x_bc < x_top_l:
            return "leftfront"
        elif x_bc < x_top_r:
            return "front"
        else:
            return "rightfront"
    else:
        return "left" if x_bc < x_bot_c else "right"

def draw_guides(draw: ImageDraw.ImageDraw, frame_w:int, frame_h:int):
    """그림과 동일한 가이드(윗부분 세로 2개, 전체 수평 1개, 아랫부분 중앙 세로 1개)"""
    x_top_l, x_top_r, x_bot_c, y_h = _zone_boundaries(frame_w, frame_h)
    w, h = frame_w, frame_h
    # 수평선(화면 전체)
    draw.line([(0, y_h), (w, y_h)], fill="gray", width=3)
    # 윗부분 세로선 2개
    draw.line([(x_top_l, 0), (x_top_l, y_h)], fill="gray", width=3)
    draw.line([(x_top_r, 0), (x_top_r, y_h)], fill="gray", width=3)
    # 아랫부분 중앙 세로선 1개
    draw.line([(x_bot_c, y_h), (x_bot_c, h)], fill="gray", width=3)

def run_interpreter():
    """image_buffer 기반 EdgeTPU 추론 후 detected_objs 갱신"""
    global detected_objs, inference_latency
    start = time.perf_counter()
    _, scale = common.set_resized_input(
        interpreter,
        image_buffer.size,
        lambda size: image_buffer.resize(size, Image.Resampling.LANCZOS)
    )
    interpreter.invoke()
    inference_latency = time.perf_counter() - start

    objs = detect.get_objects(interpreter, score_threshold=SCORE_THRESHOLD, image_scale=scale)

    # 같은 클래스에서 너무 가까운 박스(중복) 제거
    dedup = {}
    filtered = []
    for obj in objs:
        bbox = obj.bbox
        center = ((bbox.xmax + bbox.xmin) / 2.0, (bbox.ymax + bbox.ymin) / 2.0)
        bucket = dedup.get(obj.id)
        if bucket is not None:
            if any(is_duplicate(center, prev) for prev in bucket):
                continue
        else:
            dedup[obj.id] = []
        dedup[obj.id].append(center)
        filtered.append((obj, bbox))

    detected_objs = filtered

# 첫 추론 스레드
inference_thread = threading.Thread(target=run_interpreter)


# ============== 메인 루프 ==============
try:
    while True:
        frame_pil = picam2.capture_image()

        # 추론 스레드가 놀고 있으면 새 프레임으로 갱신 후 시작
        if not inference_thread.is_alive():
            image_buffer.paste(frame_pil)
            inference_thread = threading.Thread(target=run_interpreter)
            inference_thread.start()

        draw = ImageDraw.Draw(frame_pil)

        # 가이드
        if show_guides:
            draw_guides(draw, WIDTH, HEIGHT)

        # 탐지 결과 렌더링
        for obj, bbox in detected_objs:
            cls_name = labels.get(obj.id, str(obj.id))
            # 박스
            draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                           outline="yellow", width=2)

            # 하단 중앙 좌표
            x_bc = (bbox.xmin + bbox.xmax) / 2.0
            y_bc = bbox.ymax

            # 방향 판정(위3/아래2)
            direction_text = direction_from_bottom_center(x_bc, y_bc, WIDTH, HEIGHT)

            # 라벨/점수/방향
            draw.text((bbox.xmin + 10, bbox.ymin + 10),
                      f"{cls_name}\n{obj.score:.2f}\n{direction_text}",
                      fill="yellow")

            # 하단 중앙 마커
            r = 2
            draw.ellipse([(x_bc - r, y_bc - r), (x_bc + r, y_bc + r)],
                         outline="yellow", width=2)

        # FPS/지연
        draw.text((10, 10),
                  f"{int(framerate):02d} fps\n{inference_latency*1000:.2f} ms",
                  fill="white")

        # 표시
        frame_np = np.array(frame_pil)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        cv2.imshow("Food EfficientDet EdgeTPU (Top3/Bottom2 Zones)", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            show_guides = not show_guides

        # FPS 업데이트
        now = time.perf_counter()
        framerate = 1.0 / (now - last_frame_time)
        last_frame_time = now

finally:
    if inference_thread.is_alive():
        inference_thread.join()
    cv2.destroyAllWindows()
