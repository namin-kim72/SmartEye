import cv2
import torch
import numpy as np

def Capture():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.eval()

    cap = cv2.VideoCapture(0)
    print("cap.isOpened():", cap.isOpened())

    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("❌ 프레임을 읽을 수 없습니다.")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img)
        annotated = np.squeeze(results.render())

        cv2.imshow("YOLOv5n Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame

Capture()
