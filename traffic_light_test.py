import cv2
from deep_learning import PersonDetector
from config import MODEL_PATH
import numpy as np

# === 신호등 색상 감지 함수 ===
def detect_traffic_light_color(image, bbox, debug=False):
    import cv2, numpy as np

    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # ROI 살짝 축소(테두리/배경 제거)
    m = 0.12
    x1 = max(0, int(x1 + (x2-x1)*m))
    x2 = min(w-1, int(x2 - (x2-x1)*m))
    y1 = max(0, int(y1 + (y2-y1)*m))
    y2 = min(h-1, int(y2 - (y2-y1)*m))

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return "UNKNOWN"

    # 부드럽게 + HSV
    roi_blur = cv2.GaussianBlur(roi, (5,5), 0)
    hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 불빛만 통과(밝고 채도 높은 픽셀)
    s_mask = cv2.inRange(S, 100, 255)
    v_mask = cv2.inRange(V, 130, 255)
    sv = cv2.bitwise_and(s_mask, v_mask)

    # 색 마스크 (조금 보수적으로)
    red1 = cv2.inRange(hsv, (0, 80, 80),   (12, 255, 255))
    red2 = cv2.inRange(hsv, (165,80, 80),  (180,255,255))
    red  = cv2.bitwise_or(red1, red2)

    green = cv2.inRange(hsv, (55, 80, 80), (85, 255,255))

    # 불빛 마스크와 AND
    red   = cv2.bitwise_and(red, sv)
    green = cv2.bitwise_and(green, sv)

    # 노이즈 정리
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    red   = cv2.morphologyEx(red,   cv2.MORPH_OPEN, k)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)

    # 상/하 분리
    Hroi = roi.shape[0]
    top  = slice(0, Hroi//2)
    bot  = slice(Hroi//2, Hroi)

    red_top    = int(np.count_nonzero(red[top, :]))
    red_bot    = int(np.count_nonzero(red[bot, :]))
    green_top  = int(np.count_nonzero(green[top, :]))
    green_bot  = int(np.count_nonzero(green[bot, :]))
    red_sum    = red_top + red_bot
    green_sum  = green_top + green_bot

    # ROI 크기에 따른 적응형 임계값
    area = roi.shape[0] * roi.shape[1]
    PIX_MIN  = max(40, int(area * 0.002))   # 아주 작은 신호등 대비
    PIX_MIN_G = max(35, int(area * 0.0015))

    if debug:
        print(f"[DBG] red_top={red_top}, red_bot={red_bot}, green_top={green_top}, green_bot={green_bot}, "
              f"red_sum={red_sum}, green_sum={green_sum}, thrR={PIX_MIN}, thrG={PIX_MIN_G}")

    # ---------- 판정 규칙(보행 신호 특화) ----------
    # 1) 초록(하단) 우선 규칙
    if green_bot > PIX_MIN_G and green_bot > red_top * 0.7 and green_sum > red_sum * 0.5:
        return "GREEN"

    # 2) 빨강: 상단이 충분히 강하거나(보행자 빨강 사람),
    #    하단에서도 빨강(카운트 숫자)이 많고 초록이 거의 없을 때도 빨강 처리
    if (red_top > PIX_MIN) or (red_bot > PIX_MIN and green_sum < PIX_MIN_G):
        return "RED"

    # 3) 전체 빨강이 초록보다 확실히 우세
    if red_sum > green_sum * 1.8 and red_sum > PIX_MIN:
        return "RED"

    # 4) 전체 초록이 우세
    if green_sum > red_sum * 1.5 and green_sum > PIX_MIN_G:
        return "GREEN"

    return "UNKNOWN"

# === 메인 테스트 ===
def main():
    image_path = "images/traffic_4.png"  # 테스트 이미지 경로
    image = cv2.imread(image_path)

    detector = PersonDetector(model_path=MODEL_PATH, conf=0.4)
    detections = detector.detect(image)

    for bbox in detections:
        x1, y1, x2, y2, class_id, class_name = bbox
        if class_id == 9:  # traffic light
            color = detect_traffic_light_color(image, (x1, y1, x2, y2))
            print(f"[신호등 감지] 색상: {color}")

            draw_color = (0, 255, 0) if color == "GREEN" else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), draw_color, 2)
            cv2.putText(image, f"{color}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

    cv2.imshow("Traffic Light Test", cv2.resize(image, (640, 360)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
