import random

def detect_hands(frame):
    print(f"'{frame}'에서 손을 찾고 있습니다...")
    if random.choice([True, False]):
        print("프레임에서 손을 감지했습니다.")
        return {"detected": True, "position": (random.randint(50, 250), random.randint(50, 250)), "gesture": "pointing"}
    else:
        print("프레임에서 손을 찾지 못했습니다.")
        return {"detected": False, "position": None, "gesture": None}

def hands_pointing(hand_info):
    if hand_info.get("detected") and hand_info.get("gesture") == "pointing":
        print(f"손이 무언가를 가리키고 있습니다. (위치: {hand_info['position']})")
        return True
    return False
