# uv add mediapipe
import sys
import cv2
import mediapipe as mp

# mediapipe의 Hand Landmark를 추출을 위한 옵션
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)


vcap = cv2.VideoCapture(0)

while True:
    ret, frame = vcap.read()
    if not ret:
        print("카메라가 작동하지 않습니다")
        sys.exit()

    # 좌우반전
    flipped_frame = cv2.flip(frame, 1)

    ##### Hands Landmark 설정하기 #####
    # 손 그리기 설정
    frame.flags.writeable = True

    # 손 감지하기
    results = hands.process(frame)

    mylist = [5, 6, 7, 8, 9, 10, 11, 12]

    # 그리기
    if results.multi_hand_landmarks:
        print(len(results.multi_hand_landmarks))    # 손이 몇 개 탐지 되는가
        # 손 하나하나 탐색
        for hand_landmarks in results.multi_hand_landmarks:
            print(len(hand_landmarks.landmark))     # 한개의 손마다 좌표가 몇 개 나오는가? 21개

            # 손 하나의 21개의 좌표를 하나씩 출력
            height, width, _ = frame.shape

            wish_idx = [5, 6, 7, 8, 9, 10, 11, 12]
            for idx, landmark in enumerate(hand_landmarks.landmark):
                if idx in wish_idx:
                    print(f"{idx}번째 좌표 : {landmark.x}, {landmark.y}")
                    
                    point_x = int(landmark.x * width)
                    point_y = int(landmark.y * height)

                    cv2.circle(frame, (point_x, point_y), 5, (0, 0, 255), 2)

                # print(hand_landmarks.landmark)
                
                # height, width, _ = frame.shape  # (h, w, c)

                # for landmark in hand_landmarks.landmark:
                #     print(landmark.x, landmark.y)

                # point_x = int(landmark.x * width)
                # point_y = int(landmark.y * height)

                # cv2.circle(frame, (point_x, point_y), 5, (0, 0, 255), 2)


            # 자동그리기
            # mp_drawing.draw_landmarks(
            #     frame,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style()
            # )
    
    ##################################
    # 색 반전
    contrast_frame = 255 - flipped_frame

    # 화면띄우기
    cv2.imshow("webcam", frame)
    # cv2.imshow("webcam", flipped_frame)
    # cv2.imshow("webcam", contrast_frame)

    # 종료 조건
    key = cv2.waitKey(1)
    if key == 27:
        break
vcap.release()
cv2.destroyAllWindows()