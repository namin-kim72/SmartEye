import cv2
from deep_learning import PersonDetector # 모델
from img_processing import enhance_image  # 전처리 함수
from config import MODEL_PATH
import numpy as np

# === 신호등 색상 감지 함수 ===
def detect_traffic_light_color(image, bbox):
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 빨간색 HSV 범위
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 파란불(청록색) 범위
    lower_green = np.array([90, 100, 100])
    upper_green = np.array([140, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels > green_pixels and red_pixels > 100:
        return "RED"
    elif green_pixels > red_pixels and green_pixels > 100:
        return "GREEN"
    else:
        return "UNKNOWN"

# === 메인 테스트 함수 ===
def main():
    cap = cv2.VideoCapture("traffic_light.mp4")
    detector = PersonDetector(model_path=MODEL_PATH, conf=0.4)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        enhanced = enhance_image(frame)  # 영상 전처리
        detections = detector.detect(enhanced)

        for bbox in detections:
            x1, y1, x2, y2, class_id, class_name = bbox

            # 신호등(class_id=9)만 분석
            if class_id == 9:
                color = detect_traffic_light_color(enhanced, (x1, y1, x2, y2))
                print(f"[신호등 감지] 색상: {color}")

                # 시각화
                color_draw = (0, 255, 0) if color == "GREEN" else (0, 0, 255)
                cv2.rectangle(enhanced, (x1, y1), (x2, y2), color_draw, 2)
                cv2.putText(enhanced, f"{color}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_draw, 2)

        cv2.imshow("Traffic Light Detection", cv2.resize(enhanced, (640, 360)))

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
