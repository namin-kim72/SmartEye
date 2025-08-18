# WalkingMode/roadFriend.py
import cv2

from Common.capture import Capture
from Common.img_processing import DistanceEstimator, detect_traffic_light_color
from WalkingMode.deep_learning import PersonDetector
from WalkingMode.risk import RiskClassifier, RiskLevel
from WalkingMode.notifier import Notifier
from WalkingMode.config import WINDOW_TITLE, DISPLAY_SIZE

def RoadFriend():
    detector = PersonDetector(conf=None)  # conf는 config에서 기본 사용
    estimator = DistanceEstimator()
    classifier = RiskClassifier()
    notifier = Notifier()

    while True:
        frame_rgb = Capture()
        if frame_rgb is None:
            continue

        # OpenCV 색공간 일치(BGR) — deep_learning, traffic_light 모두 BGR 기준
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 객체 감지
        detections = detector.detect(frame)

        # 거리 추정 + 위험 판정
        distances_mm = []
        for (x1, y1, x2, y2, cx, cy, class_id, class_name) in detections:
            if class_id == 9:  # traffic light은 거리 제외
                distances_mm.append(-1)
                continue
            dist_mm = estimator.estimate_from_bbox((x1, y1, x2, y2, class_id, class_name),
                                                   image_height=frame.shape[0])
            risk = classifier.classify(dist_mm, class_id=class_id)
            notifier.notify(risk)
            distances_mm.append(dist_mm)
            print(f"[INFO] {class_name}: {dist_mm/1000:.2f} m, 위험도: {risk}")

        # 시각화
        for i, (x1, y1, x2, y2, cx, cy, class_id, class_name) in enumerate(detections):
            d_mm = distances_mm[i]
            if class_id == 9:
                color_name = detect_traffic_light_color(frame, (x1, y1, x2, y2, class_id, class_name))
                color = (0, 255, 0) if color_name == "GREEN" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{color_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                continue

            # 위험도 색
            if d_mm < 0:
                color = (0, 255, 0)
                text = f"{class_name} --"
            else:
                if classifier.classify(d_mm, class_id) == RiskLevel.DANGER:
                    color = (0, 0, 255)
                elif classifier.classify(d_mm, class_id) == RiskLevel.CAUTION:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)
                text = f"{class_name} {d_mm/1000:.2f}m"

            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 디스플레이
        disp = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW_TITLE, disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
def main() :
    RoadFriend()
    
if __name__ == "__main__":
    main()
