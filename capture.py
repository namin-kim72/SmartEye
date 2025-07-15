import cv2

cap = cv2.VideoCapture(0)

def Capture():
  ret, frame = cap.read()
  if not ret:
    print("Capture 이미지 실패 ")
    return None
  return frame