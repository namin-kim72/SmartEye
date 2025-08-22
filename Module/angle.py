# mpu6050_thread.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라즈베리파이 + MPU-6050 상보필터 스레드 런너
- 메인 스레드에서는 함수 몇 개만 호출하면 됩니다:
    t = imu_start(hz=100, alpha=0.98, bus_num=1, addr=0x68, callback=None, calibrate_sec=1.0)
    pitch, roll = imu_get()
    imu_stop()
- callback(pitch, roll, ax, ay, az, gx, gy, dt) 를 넘기면 측정마다 호출됩니다.
- I2C 배선: VCC→3.3V, GND→GND, SDA→GPIO2(핀3), SCL→GPIO3(핀5)
"""

import time
import math
import atexit
import threading
from typing import Optional, Callable, Tuple

import smbus2

# -------------------------
# 레지스터 / 주소
# -------------------------
MPU6050_ADDR   = 0x68
PWR_MGMT_1     = 0x6B
ACCEL_XOUT_H   = 0x3B
ACCEL_YOUT_H   = 0x3D
ACCEL_ZOUT_H   = 0x3F
GYRO_XOUT_H    = 0x43
GYRO_YOUT_H    = 0x45
GYRO_ZOUT_H    = 0x47

ACCEL_SENS     = 16384.0   # ±2g
GYRO_SENS      = 131.0     # ±250 dps

# -------------------------
# 내부 상태
# -------------------------
__bus: Optional[smbus2.SMBus] = None
__bus_num: int = 1
__addr: int = MPU6050_ADDR

__thread: Optional[threading.Thread] = None
__stop_evt = threading.Event()
__lock = threading.Lock()

__pitch: float = 0.0
__roll: float = 0.0
__running: bool = False

# -------------------------
# 저수준 유틸
# -------------------------
def __read_word(bus: smbus2.SMBus, addr: int, reg: int) -> int:
    hi = bus.read_byte_data(addr, reg)
    lo = bus.read_byte_data(addr, reg + 1)
    val = (hi << 8) | lo
    return val - 65536 if val >= 0x8000 else val

def __init_device(bus_num: int, addr: int) -> smbus2.SMBus:
    bus = smbus2.SMBus(bus_num)
    # 슬립 해제
    bus.write_byte_data(addr, PWR_MGMT_1, 0x00)
    time.sleep(0.05)
    return bus

def __calibrate_gyro(bus: smbus2.SMBus, addr: int, sec: float = 1.0) -> Tuple[float, float, float]:
    """자이로 오프셋(편향) 추정: 평평한 곳에 고정해두고 평균."""
    n = max(1, int(sec * 200))
    sx = sy = sz = 0.0
    for _ in range(n):
        gx = __read_word(bus, addr, GYRO_XOUT_H) / GYRO_SENS
        gy = __read_word(bus, addr, GYRO_YOUT_H) / GYRO_SENS
        gz = __read_word(bus, addr, GYRO_ZOUT_H) / GYRO_SENS
        sx += gx; sy += gy; sz += gz
        time.sleep(0.005)
    return sx/n, sy/n, sz/n

# -------------------------
# 워커 스레드
# -------------------------
def __worker(hz: int, alpha: float, bus_num: int, addr: int,
             callback: Optional[Callable[[float,float,float,float,float,float,float,float], None]],
             calibrate_sec: float):
    global __bus, __pitch, __roll, __running
    try:
        __bus = __init_device(bus_num, addr)

        # 초기 각도(가속도계 기반)
        ax = __read_word(__bus, addr, ACCEL_XOUT_H) / ACCEL_SENS
        ay = __read_word(__bus, addr, ACCEL_YOUT_H) / ACCEL_SENS
        az = __read_word(__bus, addr, ACCEL_ZOUT_H) / ACCEL_SENS
        roll_accel  = math.degrees(math.atan2(ay, math.sqrt(ax*ax + az*az)))
        pitch_accel = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))
        with __lock:
            __roll = roll_accel
            __pitch = pitch_accel

        # 자이로 오프셋 보정
        gx_b, gy_b, _ = __calibrate_gyro(__bus, addr, sec=calibrate_sec)

        # 루프
        __stop_evt.clear()
        __running = True
        dt_target = 1.0 / max(1, hz)
        t_prev = time.perf_counter()

        while not __stop_evt.is_set():
            t_now = time.perf_counter()
            dt = max(1e-4, t_now - t_prev)
            t_prev = t_now

            # 가속도계
            ax = __read_word(__bus, addr, ACCEL_XOUT_H) / ACCEL_SENS
            ay = __read_word(__bus, addr, ACCEL_YOUT_H) / ACCEL_SENS
            az = __read_word(__bus, addr, ACCEL_ZOUT_H) / ACCEL_SENS

            # 자이로 (dps) - 오프셋 제거
            gx = __read_word(__bus, addr, GYRO_XOUT_H) / GYRO_SENS - gx_b
            gy = __read_word(__bus, addr, GYRO_YOUT_H) / GYRO_SENS - gy_b

            # 가속도계 각도
            roll_accel  = math.degrees(math.atan2(ay, math.sqrt(ax*ax + az*az)))
            pitch_accel = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))

            # 상보 필터
            # 관례: roll은 x축 회전(gyro_x), pitch는 y축 회전(gyro_y) 사용
            with __lock:
                __roll  = alpha * (__roll  + gx * dt) + (1.0 - alpha) * roll_accel
                __pitch = alpha * (__pitch - gy * dt) + (1.0 - alpha) * pitch_accel
                pr = (__pitch, __roll)

            if callback:
                try:
                    callback(pr[0], pr[1], ax, ay, az, gx, gy, dt)
                except Exception:
                    pass

            # 타이밍 맞추기
            rem = dt_target - (time.perf_counter() - t_now)
            if rem > 0:
                time.sleep(rem)

    finally:
        __running = False
        try:
            if __bus is not None:
                __bus.close()
        except Exception:
            pass
        __bus = None

# -------------------------
# 공개 API
# -------------------------
def imu_start(hz: int = 100,
              alpha: float = 0.98,
              bus_num: int = 1,
              addr: int = MPU6050_ADDR,
              callback: Optional[Callable[[float,float,float,float,float,float,float,float], None]] = None,
              calibrate_sec: float = 1.0) -> threading.Thread:
    """
    MPU-6050 백그라운드 측정을 시작합니다(비차단).
    반환: Thread (필요 시 .join() 가능)
    """
    global __thread, __bus_num, __addr
    if __thread and __thread.is_alive():
        return __thread
    __bus_num, __addr = bus_num, addr
    __stop_evt.clear()
    __thread = threading.Thread(
        target=__worker,
        args=(hz, alpha, bus_num, addr, callback, calibrate_sec),
        daemon=True
    )
    __thread.start()
    return __thread

def imu_stop(timeout: Optional[float] = 2.0) -> None:
    """백그라운드 측정을 중단하고 I2C를 정리합니다."""
    global __thread
    __stop_evt.set()
    if __thread and __thread.is_alive():
        __thread.join(timeout=timeout)
    __thread = None

def imu_get() -> Tuple[float, float]:
    """현재 추정된 (pitch_deg, roll_deg) 를 반환합니다."""
    with __lock:
        return __pitch, __roll

def imu_running() -> bool:
    """백그라운드 측정 스레드 동작 여부."""
    return __running

@atexit.register
def __cleanup():
    try:
        imu_stop()
    except Exception:
        pass
