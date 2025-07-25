from capture import Capture
from deep_learning import PersonDetector
from img_processing import enhance_image, DistanceEstimator
from risk import RiskClassifier
from notifier import Notifier
from config import MODEL_PATH
import cv2
def main():
    # 모델 및 거리 추정기 초기화
    detector = PersonDetector(model_path=MODEL_PATH, conf=0.4)
    estimator = DistanceEstimator()
    classifier = RiskClassifier()
    notifier = Notifier()
    while True:
        frame = Capture()
        if frame is None:
            continue

        # 전처리
        enhanced = enhance_image(frame)

        # 사람 감지
        persons = detector.detect(enhanced)

        # 거리 추정
        distances = []
        for bbox in persons:
            try:
                result = estimator.estimate_from_bbox(bbox)
                distance = result['distance_m']
                risk = classifier.classify(distance)
                notifier.notify(risk)
                distances.append(distance)
                print(f"[INFO] 거리: {distance:.2f}m, 위험도: {risk}")
            except Exception as e:
                print(f"[에러] 거리 계산 실패: {e}")
                distances.append(-1)

        # 시각화
        for i, (x1, y1, x2, y2, class_id, class_name) in enumerate(persons):

            d_m = distances[i]
            color = (0, 255, 0)  # 기본: 초록

            if d_m < 1.0:
                color = (0, 0, 255)  # 위험: 빨강
            elif d_m < 2.0:
                color = (0, 255, 255)  # 주의: 노랑

            cv2.rectangle(enhanced, (x1, y1), (x2, y2), color, 2)
            cv2.putText(enhanced, f"{class_name} {d_m:.2f}m", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 디스플레이
        disp = cv2.resize(enhanced, (640, 320), interpolation=cv2.INTER_AREA)
        cv2.imshow("YOLOv5 Detection", disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()
