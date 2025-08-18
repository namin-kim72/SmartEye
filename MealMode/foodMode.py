#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, sys, threading, time
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw

from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

# --- 모드 전용 설정 ---
from MealMode.config import (
    SCORE_THRESHOLD, DEDUP_DISTANCE,
    WINDOW_TITLE, SHOW_WINDOW, SHOW_FPS, FONT_COLOR, BOX_COLOR, DISPLAY_SIZE,
    MEAL_MODEL_PATH, MEAL_LABEL_PATH
)

# --- 공용 캡처 ---
from Common.capture import Capture

# --------------------------------------------------------------------------
# 모델/라벨 로드
# --------------------------------------------------------------------------
script_dir = Path(__file__).parent
labels = read_label_file(str(script_dir / MEAL_LABEL_PATH))
interpreter = make_interpreter(str(script_dir / MEAL_MODEL_PATH))
interpreter.allocate_tensors()

# --------------------------------------------------------------------------
# 구역/방향(테스트 스크립트 로직 이식)
# --------------------------------------------------------------------------
# 위쪽 3분할(좌전/전방/우전), 아래쪽 2분할(좌/우)
H_SPLIT   = 0.50      # 수평 분할 위치(화면 높이 비율)
TOP_L     = 1.0/3.0   # 윗부분 좌/전 경계(가로 비율)
TOP_R     = 2.0/3.0   # 윗부분 전/우 경계(가로 비율)
BOTTOM_C  = 0.50      # 아랫부분 좌/우 경계(가로 비율)

def _zone_boundaries(w:int, h:int):
    y_h = int(h * H_SPLIT)
    x_top_l = int(w * TOP_L)
    x_top_r = int(w * TOP_R)
    x_bot_c = int(w * BOTTOM_C)
    return x_top_l, x_top_r, x_bot_c, y_h

def direction_from_bottom_center(x_bc: float, y_bc: float,
                                 frame_w: int, frame_h: int) -> str:
    x_top_l, x_top_r, x_bot_c, y_h = _zone_boundaries(frame_w, frame_h)
    if y_bc <= y_h:
        if x_bc < x_top_l:   return "leftfront"
        elif x_bc < x_top_r: return "front"
        else:                return "rightfront"
    else:
        return "left" if x_bc < x_bot_c else "right"

def draw_guides(draw: ImageDraw.ImageDraw, frame_w:int, frame_h:int):
    x_top_l, x_top_r, x_bot_c, y_h = _zone_boundaries(frame_w, frame_h)
    w, h = frame_w, frame_h
    draw.line([(0, y_h), (w, y_h)], fill="gray", width=2)             # 수평선
    draw.line([(x_top_l, 0), (x_top_l, y_h)], fill="gray", width=2)   # 윗부분 좌/전
    draw.line([(x_top_r, 0), (x_top_r, y_h)], fill="gray", width=2)   # 윗부분 전/우
    draw.line([(x_bot_c, y_h), (x_bot_c, h)], fill="gray", width=2)   # 아랫부분 좌/우

# --------------------------------------------------------------------------
# 중복 제거 유틸
# --------------------------------------------------------------------------
def is_duplicate(center1, center2, dist_thresh=DEDUP_DISTANCE):
    return dist_thresh >= math.dist(center1, center2)

# --------------------------------------------------------------------------
# 추론 스레드
# --------------------------------------------------------------------------
image_buffer = None
detected_objs = []        # [(obj, bbox), ...]
inference_latency = sys.float_info.max
_infer_lock = threading.Lock()

def run_interpreter():
    global image_buffer, detected_objs, inference_latency
    if image_buffer is None:
        return
    start = time.perf_counter()

    # PIL 이미지 버퍼를 모델 입력 크기에 맞춰 리사이즈
    _, scale = common.set_resized_input(
        interpreter,
        image_buffer.size,
        lambda size: image_buffer.resize(size, Image.Resampling.LANCZOS),
    )
    interpreter.invoke()
    inference_latency = time.perf_counter() - start

    objs = detect.get_objects(interpreter, score_threshold=SCORE_THRESHOLD, image_scale=scale)

    # 같은 클래스에서 너무 가까운 박스(중복) 제거
    dedup_map, filtered = {}, []
    for obj in objs:
        bbox = obj.bbox
        center = ((bbox.xmax + bbox.xmin) / 2, (bbox.ymax + bbox.ymin) / 2)

        bucket = dedup_map.get(obj.id)
        if bucket is not None:
            if any(is_duplicate(center, oc) for oc in bucket):
                continue
        else:
            dedup_map[obj.id] = []
        dedup_map[obj.id].append(center)
        filtered.append((obj, bbox))

    with _infer_lock:
        detected_objs = filtered

# --------------------------------------------------------------------------
# 메인 루프
# --------------------------------------------------------------------------
def main():
    global image_buffer
    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    inference_thread = threading.Thread(target=run_interpreter)
    inference_thread.start()

    framerate = 0.0
    last_frame_time = time.perf_counter()
    show_guides = False

    try:
        while True:
            frame_rgb = Capture()
            if frame_rgb is None:
                continue

            # PIL.Image로 변환(여기서 그리기 수행)
            frame_pil = Image.fromarray(frame_rgb)

            # 추론 스레드 갱신
            if not inference_thread.is_alive():
                image_buffer = frame_pil.copy()
                inference_thread = threading.Thread(target=run_interpreter)
                inference_thread.start()

            draw = ImageDraw.Draw(frame_pil)
            w, h = frame_pil.size

            # 가이드선 표시
            if show_guides:
                draw_guides(draw, w, h)

            # 탐지 결과 렌더링 + 방향 표기(테스트 스크립트 로직)
            with _infer_lock:
                current = list(detected_objs)

            for obj, bbox in current:
                cls_name = labels.get(obj.id, str(obj.id))

                # 박스
                draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                               outline="yellow", width=2)

                # 하단 중앙 좌표
                x_bc = (bbox.xmin + bbox.xmax) / 2.0
                y_bc = bbox.ymax

                # 방향 판정(위3/아래2)
                direction_text = direction_from_bottom_center(x_bc, y_bc, w, h)

                # 라벨/점수/방향
                draw.text((bbox.xmin + 10, bbox.ymin + 10),
                          f"{cls_name}\n{obj.score:.2f}\n{direction_text}",
                          fill="yellow")

                # 하단 중앙 마커
                r = 2
                draw.ellipse([(x_bc - r, y_bc - r), (x_bc + r, y_bc + r)],
                             outline="yellow", width=2)

            # FPS/지연
            if SHOW_FPS:
                draw.text((10, 10), f"{int(framerate):02d} fps\n{inference_latency*1000:.2f} ms",
                          fill="white")

            # === 표시 ===
            # 요구사항: BGR<->RGB 변환 금지. 그린 결과(PIL→ndarray RGB)를 그대로 imshow.
            if SHOW_WINDOW:
                vis_rgb = np.array(frame_rgb) # RGB 그대로
                disp = cv2.resize(frame_rgb, (480, 640), interpolation=cv2.INTER_AREA)
                cv2.imshow(WINDOW_TITLE, disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('g'):
                    show_guides = not show_guides

            # FPS 갱신
            now = time.perf_counter()
            framerate = 1.0 / max(1e-6, (now - last_frame_time))
            last_frame_time = now

    finally:
        if inference_thread.is_alive():
            inference_thread.join()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()