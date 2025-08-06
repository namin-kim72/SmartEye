import cv2
from person_detector import PersonDetector
import matplotlib.pyplot as plt

def visualize_detection(image_path):
    # 모델 초기화
    detector = PersonDetector()

    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 이미지 불러오기 실패:", image_path)
        return

    # 탐지 수행
    boxes = detector.detect(image)

    # 박스 시각화
    for box in boxes:
        x1, y1, x2, y2, class_id, class_name = box
        color = (0, 255, 0)  # 초록 박스
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{class_name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 결과 출력 (matplotlib 사용 시 RGB로 변환)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("Detection Result")
    plt.show()


if __name__ == "__main__":
    # 테스트 이미지 경로 입력
    image_path = "test.jpg"
    visualize_detection(image_path)
