#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, signal, threading, time, queue
from typing import Optional

# 각 모듈 import
from WalkingMode.WalkingMode import main as run_walk_mode
from ReadMode.ReadMode import main as run_read_mode
from MealMode.foodMode import main as run_food_mode
from Common.frame_bus import FrameBus
from Common.capture import CameraManager # CameraManager 클래스 import

# ---- 그레이스풀 종료 처리 ----
def graceful_exit(*_):
    print("\n[MAIN] 종료")
    # CameraManager가 종료되면 모든 자원이 정리됨
    sys.exit(0)

# ---- 모드 선택 함수 ----
def choose(prompt: str, choices: list[str]) -> Optional[int]:
    print(prompt)
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    print("  q. 종료")
    sel = input("선택: ").strip().lower()
    if sel in ("q", "quit", "exit"):
        return None
    try:
        val = int(sel)
        if 1 <= val <= len(choices):
            return val
    except:
        pass
    print("[MAIN] 잘못된 선택")
    return choose(prompt, choices)

# ---- 메인 루프 ----
def main():
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    MENU = ["보행모드", "읽기모드", "식사모드"]
    
    # 중앙 객체들 생성 (단 하나씩만)
    frame_bus = FrameBus()
    still_capture_queue = queue.Queue()
    camera_manager = CameraManager(frame_bus, still_capture_queue)
    
    # 카메라 캡처 스레드 시작
    camera_manager.start()
    while True:
        sel = choose("\n=== SmartEye 메인 ===", MENU)
        if sel is None:
            break
        # 각 모드 함수에 bus와 camera_manager 전달
        if sel == 1:
            run_walk_mode(frame_bus)
        elif sel == 2:
            run_read_mode(frame_bus, camera_manager)
        elif sel == 3:
            run_food_mode(frame_bus)

    # 프로그램 종료 시 카메라 매니저 종료
    camera_manager.stop()

if __name__ == "__main__":
    main()
