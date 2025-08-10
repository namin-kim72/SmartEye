import cv2
import numpy as np
from deep_learning import PersonDetector  # (x1,y1,x2,y2,class_id,class_name) 반환 가정
from config import MODEL_PATH

# ------------------------------
# 신호등 색상 감지
# ------------------------------
def detect_traffic_light_color(image, bbox, debug=False):
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # 경계 보정 + 약간 패딩
    pad = 4
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0 or (y2 - y1) < 6 or (x2 - x1) < 6:
        return "UNKNOWN"

    # 전처리: 가우시안 블러 + 밝기 정규화
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)
    vch = cv2.equalizeHist(vch)
    hsv = cv2.merge([hch, sch, vch])

    # HSV 범위 (조금 넓게)
    red1_lo, red1_hi = np.array([0,   80, 80]),  np.array([12, 255, 255])
    red2_lo, red2_hi = np.array([165, 80, 80]),  np.array([180,255, 255])
    green_lo, green_hi = np.array([40, 50, 50]), np.array([95, 255, 255])

    red_mask = cv2.inRange(hsv, red1_lo, red1_hi) | cv2.inRange(hsv, red2_lo, red2_hi)
    green_mask = cv2.inRange(hsv, green_lo, green_hi)

    # 노이즈 억제
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, k)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, k, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_DILATE, k, iterations=1)

    # 수직 신호등 가정: 상/중/하로 나눠 가장 강한 구역 비교(불빛이 작은 경우 유리)
    H = roi.shape[0]
    thirds = [(0, H//3), (H//3, 2*H//3), (2*H//3, H)]
    red_scores, green_scores = [], []
    for y_lo, y_hi in thirds:
        red_scores.append(int(np.count_nonzero(red_mask[y_lo:y_hi, :])))
        green_scores.append(int(np.count_nonzero(green_mask[y_lo:y_hi, :])))

    red_sum = sum(red_scores)
    green_sum = sum(green_scores)

    if debug:
        print(f"[DEBUG] red_sum={red_sum}, green_sum={green_sum}, red_parts={red_scores}, green_parts={green_scores}")

    # 임계치(작은 신호등 대비 완화)
    RED_MIN, GREEN_MIN = 60, 40

    # 상단 램프(빨강), 하단 램프(초록) 가중치 약간 부여
    red_weight = red_sum + red_scores[0] * 0.25
    green_weight = green_sum + green_scores[-1] * 0.25

    if red_weight > green_weight and red_sum > RED_MIN:
        return "RED"
    elif green_weight >= red_weight and green_sum > GREEN_MIN:
        return "GREEN"
    else:
        return "UNKNOWN"

# ------------------------------
# 메인
# ------------------------------
def main():
    cap = cv2.VideoCapture("traffic_light.mp4")  # 파일 테스트. 웹캠이면 0
    detector = PersonDetector(model_path=MODEL_PATH, conf=0.35)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 신호등만 감지하려면 deep_learning.detect 내부에서 classes=[9] 처리 or 아래처럼 필터링
        detections = detector.detect(frame)

        for bbox in detections:
            x1, y1, x2, y2, class_id, class_name = bbox

            if class_id == 9:  # traffic light (COCO)
                color = detect_traffic_light_color(frame, (x1, y1, x2, y2), debug=False)

                draw = (0, 255, 0) if color == "GREEN" else ((0, 0, 255) if color == "RED" else (128, 128, 128))
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw, 2)
                cv2.putText(frame, f"{color}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw, 2)

        cv2.imshow("Traffic Light Detection", cv2.resize(frame, (640, 360)))
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
