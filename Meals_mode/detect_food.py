def detect_food(frame):
    print(f"'{frame}'에서 음식들을 찾습니다...")
    # 실제 음식 인식 모델을 사용하여 음식의 이름과 위치를 리스트로 반환
    food_list = [
        {"name": "제육", "box": (50, 50, 100, 100)},
        {"name": "라면", "box": (120, 150, 250, 200)},
        {"name": "돈까스", "box": (80, 220, 150, 300)}
    ]
    print(f"{len(food_list)}개의 음식을 감지했습니다: {[food['name'] for food in food_list]}")
    return food_list
