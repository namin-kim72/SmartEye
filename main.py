#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, signal
from typing import Optional

# === 모드 import ===
# 각 모드 파일에 반드시 def main(): 이 정의돼 있어야 함
try:
    from WalkingMode.WalkingMode import main as run_walk_mode
except Exception as e:
    run_walk_mode = None
    print(f"[WARN] WalkingMode import 실패: {e}")

try:
    from ReadMode.ReadMode import main as run_read_mode
except Exception as e:
    run_read_mode = None
    print(f"[WARN] ReadMode import 실패: {e}")

try:
    from MealMode.foodMode import main as run_food_mode
except Exception as e:
    run_food_mode = None
    print(f"[WARN] MealMode import 실패: {e}")


def graceful_exit(*_):
    print("\n[MAIN] 종료")
    sys.exit(0)


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


def main():
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    MENU = [
        "보행모드 (WalkingMode)",
        "읽기모드 (ReadMode)",
        "식사모드 (MealMode)",
    ]

    while True:
        sel = choose("\n=== SmartEye 메인 ===", MENU)
        if sel is None:
            graceful_exit()

        if sel == 1:
            if run_walk_mode: run_walk_mode()
            else: print("[ERR] WalkingMode 실행 불가")
        elif sel == 2:
            if run_read_mode: run_read_mode()
            else: print("[ERR] ReadMode 실행 불가")
        elif sel == 3:
            if run_food_mode: run_food_mode()
            else: print("[ERR] MealMode 실행 불가")


if __name__ == "__main__":
    main()