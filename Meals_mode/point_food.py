def find_hands():
    pass

def find_food(hand_position, food_list):
    print("손이 가리키는 음식 찾는 중...")
    if not hand_position or not food_list:
        return None

    for food in food_list:
        x1, y1, x2, y2 = food['box']
        fx, fy = hand_position
        if x1 <= fx <= x2 and y1 <= fy <= y2:
            print(f"손이 '{food['name']}'을(를) 가리키고 있습니다.")
            return food['name']
    print("손이 어떤 음식도 가리키지 않습니다.")
    return None
