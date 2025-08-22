# main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
다른 파일에서 vibrator 모듈을 불러와 메인 스레드가 계속 돌면서
진동은 별도 스레드에서 비차단으로 동작시키는 예시.
- vibrator.py 파일이 같은 폴더에 있다고 가정합니다.
- 사용법:
    python3 main.py
"""

import time
import threading
import angle
import vibrator

def on_frame(pitch, roll, ax, ay, az, gx, gy, dt):
    # 필요 시 콜백에서 실시간 처리 (로그/알람 등)
    print(f"[CB] pitch={pitch:6.2f}  roll={roll:6.2f}  dt={dt*1000:5.1f}ms")

def other_work():
    """메인 스레드에서 돌아가는 다른 작업(예시)"""
    print("(1) : 진동")
    print("(2) : 소리")
    print("(3) : 각도 확인")
    print("(4) : 거리 확인")
    time.sleep(3)

def main():
    other_work()
    while True:
        user_input = input("(input) ...")

        if user_input == "1":
            print("(1) : 0.8초 진동")
            t1 = vibrator.vibrator(800)  # 0.8초 진동, 스레드 반환
        elif user_input == "2":
            if not angle.imu_running():
                print("(2) : IMU 시작")
                # 이미 실행 중이라면 새 스레드를 만들지 않음
                t2 = angle.imu_start(hz=100, alpha=0.98, bus_num=1, addr=0x68,
                                     callback=on_frame, calibrate_sec=1.0)
            else:
                print("(2) : IMU 중지")
                angle.imu_stop()
        elif user_input == "q":
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[메인] 사용자 중단")
    # vibrator.py 안에서 atexit로 GPIO 정리 수행됨
