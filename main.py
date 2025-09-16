#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, signal, queue
from google.cloud import texttospeech as tts
import subprocess, os

from Common.config import GOOGLE_KEY_PATH
from Common.button_manager import ButtonManager
from Common.frame_bus import FrameBus
from Common.capture import CameraManager

from WalkingMode.WalkingMode import main as run_walk_mode
from ReadMode.ReadMode import main as run_read_mode
from MealMode.foodMode import main as run_food_mode
from Common.utils import graceful_exit, speak

# -------------------------
# 메인
# -------------------------

def main():
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    MENU = ["보행모드", "읽기모드", "식사모드"]

    frame_bus = FrameBus()
    still_capture_queue = queue.Queue()
    camera_manager = CameraManager(frame_bus, still_capture_queue)
    camera_manager.start()

    button_mgr = ButtonManager()

    speak("스마트아이 전원이 켜졌습니다.")

    mode_index = None

    while True:
        event = button_mgr.get_event(timeout=0.1)

        if event == "LONG_PRESS":
            graceful_exit()

        elif event == "SHORT_PRESS":
            if mode_index is None:
                mode_index = 0
            else:
                mode_index = (mode_index + 1) % len(MENU)

            speak(f"{MENU[mode_index]}가 켜졌습니다.")

            if mode_index == 0:
                run_walk_mode(frame_bus, button_mgr)
            elif mode_index == 1:
                run_read_mode(frame_bus, camera_manager, button_mgr)
            elif mode_index == 2:
                run_food_mode(frame_bus, button_mgr)

if __name__ == "__main__":
    main()

