# 카메라 접속하는 코드

def camera_connection():
    print("카메라에 접속을 시도합니다.")
    is_connected = True
    if is_connected:
        print("카메라에 성공적으로 연결되었습니다.")
        return True
    else:
        print("오류: 카메라에 연결할 수 없습니다.")
        return False

def read_frame():
    print("현재 프레임을 캡처합니다.")
    return "frame_data"
