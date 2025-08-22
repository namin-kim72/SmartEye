# vabrator.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라즈베리파이 진동 모듈 비차단 제어 (단일 API)
- import vabrator 후 vabrator.vibrator(ms) 한 함수만 호출하면 됩니다.
- 내부적으로 스레드에서 동작하며, 겹쳐 호출 시 참조 카운트로 HIGH/LOW를 안전하게 관리합니다.
"""

import time
import threading
import atexit
from typing import Optional

import RPi.GPIO as GPIO

# 내부 상태
_PIN_DEFAULT = 17
__pin_in_use: Optional[int] = None
__initialized = False
__lock = threading.Lock()
__ref_count = 0  # 동시에 여러 요청이 들어올 때를 위한 카운터


def __ensure_setup(pin: int):
    """GPIO 초기화(한 번만). 다른 핀 요구 시 안전하게 전환."""
    global __initialized, __pin_in_use
    if not __initialized:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        __initialized = True
        __pin_in_use = pin
        return

    if __pin_in_use != pin:
        # 기존 핀 LOW로 내리고 새 핀 준비
        try:
            GPIO.output(__pin_in_use, GPIO.LOW)
        except Exception:
            pass
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        __pin_in_use = pin


def __pulse_worker(duration_ms: int, pin: int):
    """비차단 진동 스레드 본체."""
    global __ref_count
    __ensure_setup(pin)

    try:
        # 시작: 참조 카운트 증가, 첫 시작이면 HIGH
        with __lock:
            first = (__ref_count == 0)
            __ref_count += 1
        if first:
            GPIO.output(pin, GPIO.HIGH)

        # 지정 시간 유지
        time.sleep(max(0, duration_ms) / 1000.0)

    finally:
        # 종료: 참조 카운트 감소, 마지막이면 LOW
        with __lock:
            __ref_count = max(0, __ref_count - 1)
            last = (__ref_count == 0)
        if last:
            GPIO.output(pin, GPIO.LOW)


def vibrator(ms: int, pin: int = _PIN_DEFAULT) -> threading.Thread:
    """
    ms(밀리초) 동안 진동. 비차단(독립 스레드)으로 동작.
    예) import vabrator; vabrator.vibrator(1200)  # 1.2초 진동
    반환값: Thread 객체(필요 시 .join()으로 대기 가능)
    """
    t = threading.Thread(target=__pulse_worker, args=(ms, pin), daemon=True)
    t.start()
    return t


@atexit.register
def __cleanup():
    """프로세스 종료 시 안전하게 LOW 및 GPIO 정리."""
    global __initialized, __pin_in_use, __ref_count
    try:
        if __initialized and __pin_in_use is not None:
            GPIO.output(__pin_in_use, GPIO.LOW)
        GPIO.cleanup()
    except Exception:
        pass
    finally:
        __initialized = False
        __pin_in_use = None
        __ref_count = 0
