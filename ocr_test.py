from PIL import Image
import pytesseract

# 설치 경로에 맞게 경로 지정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = Image.open(r'C:\Users\FORYOUCOM\Desktop\공학경진대회\sample_image2.png')  # 테스트할 이미지 파일명

# 한글+영어 OCR (kor+eng)
result = pytesseract.image_to_string(img, lang='kor+eng')
print(result)
