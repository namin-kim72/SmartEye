#!/usr/bin/python3
# -*- coding: utf-8 -*-

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


# =========================
# 기본 설정
# =========================
# 파일 시스템 경로
script_path = Path(__file__)
script_dir = script_path.parent

# 카메라 해상도 (세로형: 480x640)
WIDTH, HEIGHT = 480, 640

# 방향 구간 비율 (필요 시 조정)
CENTER_BAND = 0.20  # 화면 가로의 가운데 '전방' 폭 비율 (기본 20%)
FLANK_BAND  = 0.20  # 전방 양옆 '좌측전방/우측전방' 폭 비율 (기본 각 20%)

# 탐지 임계치 및 중복 제거 파라미터
SCORE_THRESHOLD = 0.15
DEDUP_CENTER_DIST = 15  # 같은 클래스에서 중심 간 거리가 이 값 이하이면 중복으로 간주


# =========================
# 카메라 설정
# =========================
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


# =========================
# 모델 / 라벨 로드 (EfficientDet-Lite0 EdgeTPU)
# =========================
# labels: 라인별 클래스명 (예: ./model/labels.txt)
# model : EdgeTPU용 양자화 tflite (예: ./model/food_effi_lite0_200ep_int8.tflite)
image_buffer = Image.new("RGB", (WIDTH, HEIGHT))
labels = read_label_file(str(script_dir / "./model/labels.txt"))

interpreter = make_interpreter(
    str(script_dir / "./model/food_effi_lite0_200ep_int8.tflite")
)
interpreter.allocate_tensors()


# =========================
# 전역 상태
# =========================
detected_objs = []                 # [(obj, bbox), ...]
inference_latency = sys.float_info.max
last_frame_time = time.perf_counter()
framerate = 0.0
show_guides = False                # 가이드라인 표시 여부 토글


# =========================
# 유틸 함수
# =========================
def is_duplicate(center1, center2, dist_thresh=DEDUP_CENTER_DIST) -> bool:
    """두 중심점 사이가 dist_thresh 이하면 중복으로 판단."""
    return dist_thresh >= math.dist(center1, center2)


def direction_from_bottom_center(x_bc: float, frame_w: int,
                                 center_band: float = CENTER_BAND,
                                 flank_band: float = FLANK_BAND) -> str:
    """
    하단 중앙 x좌표(x_bc)와 프레임 폭(frame_w)을 받아 5방향 텍스트를 반환.

    구간 (0 ~ W):
      [0, L)            -> 좌측
      [L, LF)           -> 좌측전방
      [LF, RF]          -> 전방
      (RF, R]           -> 우측전방
      (R, W]            -> 우측

    비율 파라미터:
      - center_band: 가운데 '전방' 폭(비율)
      - flank_band : 전방 양옆 '전방측' 폭(비율)
    """
    w = float(frame_w)

    # 경계선 계산
    left_edge = w * flank_band
    left_front_edge = w * (flank_band + center_band / 2.0)
    right_front_edge = w * (1.0 - (flank_band + center_band / 2.0))
    right_edge = w * (1.0 - flank_band)

    if x_bc < left_edge:
        return "좌측"
    elif x_bc < left_front_edge:
        return "좌측전방"
    elif x_bc <= right_front_edge:
        return "전방"
    elif x_bc <= right_edge:
        return "우측전방"
    else:
        return "우측"


def run_interpreter():
    """image_buffer 기반으로 EdgeTPU 추론을 수행하고 detected_objs 갱신."""
    global detected_objs, inference_latency

    start = time.perf_counter()
    # set_resized_input: 입력 리사이즈 및 scale 반환
    _, scale = common.set_resized_input(
        interpreter,
        image_buffer.size,
        lambda size: image_buffer.resize(size, Image.Resampling.LANCZOS)
    )
    interpreter.invoke()
    inference_latency = time.perf_counter() - start

    # EfficientDet 결과 파싱
    objs = detect.get_objects(
        interpreter,
        score_threshold=SCORE_THRESHOLD,
        image_scale=scale
    )

    # 같은 클래스 내에서 너무 가까운(중복) 박스 제거
    dedup_map = {}  # cls_id -> [centers...]
    filtered = []

    for obj in objs:
        bbox = obj.bbox
        center = ((bbox.xmax + bbox.xmin) / 2.0, (bbox.ymax + bbox.ymin) / 2.0)

        bucket = dedup_map.get(obj.id)
        if bucket is not None:
            if any(is_duplicate(center, prev) for prev in bucket):
                continue
        else:
            dedup_map[obj.id] = []

        dedup_map[obj.id].append(center)
        filtered.append((obj, bbox))

    detected_objs = filtered


# 첫 추론 스레드 준비
inference_thread = threading.Thread(target=run_interpreter)


# =========================
# 메인 루프
# =========================
try:
    while True:
        # Picamera2 → PIL.Image (RGB888)
        frame_pil = picam2.capture_image()

        # 추론 스레드가 비어 있으면 새 프레임으로 갱신 후 시작
        if not inference_thread.is_alive():
            image_buffer.paste(frame_pil)
            inference_thread = threading.Thread(target=run_interpreter)
            inference_thread.start()

        # 드로잉 컨텍스트
        draw = ImageDraw.Draw(frame_pil)

        # (옵션) 가이드 라인: 5구역 시각화
        if show_guides:
            w = float(WIDTH)
            left_edge = w * FLANK_BAND
            left_front_edge = w * (FLANK_BAND + CENTER_BAND / 2.0)
            right_front_edge = w * (1.0 - (FLANK_BAND + CENTER_BAND / 2.0))
            right_edge = w * (1.0 - FLANK_BAND)
            guide_xs = [left_edge, left_front_edge, right_front_edge, right_edge]
            for gx in guide_xs:
                draw.line([(gx, 0), (gx, HEIGHT)], fill="gray", width=1)

        # 탐지 결과 그리기
        for obj, bbox in detected_objs:
            cls_name = labels.get(obj.id, str(obj.id))

            # 바운딩박스
            draw.rectangle(
                [(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                outline="yellow",
                width=2
            )

            # === 하단 중앙 좌표 ===
            x_bc = (bbox.xmin + bbox.xmax) / 2.0
            y_bc = bbox.ymax

            # === 방향 텍스트 판정 ===
            direction_text = direction_from_bottom_center(x_bc, WIDTH)

            # 라벨/점수/방향 표시
            draw.text(
                (bbox.xmin + 10, bbox.ymin + 10),
                f"{cls_name}\n{obj.score:.2f}\n{direction_text}",
                fill="yellow"
            )

            # 하단 중앙 마커(디버깅용)
            r = 2
            draw.ellipse(
                [(x_bc - r, y_bc - r), (x_bc + r, y_bc + r)],
                outline="yellow",
                width=2
            )

        # FPS / 추론 지연(ms)
        draw.text(
            (10, 10),
            f"{int(framerate):02d} fps\n{inference_latency*1000:.2f} ms",
            fill="white"
        )

        # PIL → OpenCV(BGR) 변환 후 화면 표시
        frame_np = np.array(frame_pil)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        cv2.imshow("Food EfficientDet EdgeTPU", frame_bgr)

        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("g"):  # 가이드 토글
            show_guides = not show_guides

        # FPS 계산
        this_frame_time = time.perf_counter()
        framerate = 1.0 / (this_frame_time - last_frame_time)
        last_frame_time = this_frame_time

finally:
    if inference_thread.is_alive():
        inference_thread.join()
    cv2.destroyAllWindows()
