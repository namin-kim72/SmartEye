from risk import RiskLevel

class Notifier:
    def notify(self, risk_level):
        if risk_level == RiskLevel.DANGER:
            self._vibrate()
            self._beep()
            print("🔴 위험: 진동 + 경고음 발생")
        elif risk_level == RiskLevel.CAUTION:
            self._beep()
            print("🟡 주의: 경고음 발생")
        else:
            print("🟢 안전")

    def _vibrate(self):
        # TODO: 하드웨어 연동 시 GPIO나 시리얼 명령 넣기
        print("[진동 모듈 작동]")

    def _beep(self):
        # TODO: 사운드 모듈 연동 (예: PWM, GPIO, Piezo)
        print("[경고음 발생]")
