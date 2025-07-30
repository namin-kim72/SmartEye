import camera
import detect_food
import detect_hands
import point_food
import read_food

def main():
    # 카메라 접속
    if not camera.camera_connection():
        print("프로그램을 종료합니다.")
        return

    try:
        while True:
            # 카메라 프레임 받아오기
            frame = camera.read_frame()

            # 프레임 내의 손 인식
            hand_info = detect_hands.detect_hands(frame)


            # 만약 손이 무언가를 가르키고 있다면
            if detect_hands.hands_pointing(hand_info) == True:

               # 손이 가르키는 음식 인식
               food_data = detect_food.detect_food(frame)

               food_to_read = point_food.find_food(hand_info["position"], food_data)

               # 인식된 음식 읽어주기
               read_food.read_food(food_to_read)

    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")

if __name__ == '__main__':
    main()

"""
위 코드는 반드시 이렇게 진행되어야 하는 것은 아님

카메라 접속

카메라 프레임 받아오기

딥러닝 모델로 프레임 내의 손 인식

손이 음식을 가르키고 있다면

    딥러닝 모델로 손이 가르키는 음식을 인식
    TTS로 어떤 음식인지 읽어주기
    
    
자세한 파이프라인 없이 print() 구문으로 전체 흐름도를 완성
"""
