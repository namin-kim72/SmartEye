import camera
import detect_hands
import detect_text
import read_text
import point_text

def main():
    # 카메라 접속
    camera.camera_connection()

    while True:
        # 카메라 프레임 받아오기
        camera.read_frame()

        # 프레임 내의 손 인식
        detect_hands.detect_hands()

        # 만약 손이 무언가를 가르키고 있다면
        if detect_hands.hands_pointing() == True:
            # 손이 가르키는 문장 인식
            detect_text.detect_text()
            point_text.find_text()

            # 인식된 문장 읽어주기
            read_text.read_text()

if __name__ == '__main__':
    main()

"""
위 코드는 반드시 이렇게 진행되어야 하는 것은 아님

카메라 접속

카메라 프레임 받아오기

딥러닝 모델로 프레임 내의 손 인식

손이 문장을 가르키고 있다면

    딥러닝 모델로 손이 가르키는 문장 인식
    TTS로 문장 읽어주기
    
    
자세한 파이프라인 없이 print() 구문으로 전체 흐름도를 완성
"""