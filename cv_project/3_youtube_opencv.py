# uv add yt_dlp
# uv add ultralytics

from ultralytics import YOLO
import yt_dlp
import cv2

# YOLO 모델 불러오기
model = YOLO("yolo11n.pt")

youtube_url = "https://youtu.be/S5nsDT5oU90"

# yt_dlp 옵션 설정
ydl_opts = {
    "format" : "best[ext=mp4][protocol=https]/best",
    "quite" : True,
    "no_warnings" : True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(youtube_url, download=False)
    stream_url = info_dict["url"]

vcap = cv2.VideoCapture(stream_url)

while True:
    if not vcap.isOpened():
        print("비디오를 열 수 없습니다.")
        break
    
    ret, frame = vcap.read()
    if not ret:
        print("비디오 프레임을 읽을 수 없습니다.")
        break

    # YOLO 예측하기
    results = model(frame)
    result = results[0]
    boxes = result.boxes
    # print(boxes.data)   # 박스좌표, conf, cls

    # Box 그리기
    cnt = 0
    for x1, y1, x2, y2, conf, cls_idx in boxes.data:    # cls: python 예약어 이므로 cls_idx로 대체
        # person만 박스 그리기 위한 조건
        if cls_idx > 0:
            continue

        cnt += 1

        # 좌표를 정수로 변환
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Box 그리기 (frame, (x1, y1), (x2, y2), 색상, 두께, 기타옵션)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 3)

    # 텍스트 작성 (frame, 텍스트, 위치, 폰트, 크기, 색상, 두께)
    cnt_text = f"People Count : {cnt}"
    cv2.putText(frame, cnt_text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)

    # 프레임 띄우기
    cv2.imshow("Youtube Video", frame)

    # 종료 조건
    key = cv2.waitKey(1)
    if key == 27:       # ESC 버튼 누르면 종료
        break

vcap.release()
cv2.destroyAllWindows()