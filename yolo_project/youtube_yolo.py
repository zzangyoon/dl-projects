from ultralytics import YOLO

# 모델 불러오기
model = YOLO("yolo11n.pt")

# 유튜브 url 가져오기
youtube_url = "https://youtu.be/S5nsDT5oU90"

# 예측하기
results = model(youtube_url, stream=True, show=True)

# 결과 출력해보기
for res in results:
    print(res.boxes.cls)