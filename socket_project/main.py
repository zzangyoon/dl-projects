# uv add "fastapi[all]"
# 서버 실행 : uvicorn main:app --port 8000 --reload
# (주의) 서버 실행시 경로가 socket_project 여야함
from fastapi import FastAPI, WebSocket
from models.chatgpt import mychat

app = FastAPI()

@app.get("/")
def home():
    return {"hello": "world"}

# 웹소켓
## 1. (언리얼) 텍스트 <---> (AI) 텍스트 생성     | chatgpt
## 2. (언리얼) 이미지 <---> (AI) JSON 준다      | mediapipe
## 3. (언리얼) .wav파일 <---> (AI) 텍스트 생성   | stt결과

@app.websocket("/ws/streaming")
async def websocket_text(websocket: WebSocket):
    await websocket.accept()

    try:
        # 데이터 text로 받기
        # data = await websocket.receive_text()
        # 데이터 json으로 받기
        data = await websocket.receive_json()
        print(data, type(data), data["question"])

        # AI 활동
        # response = ["안녕", "반가워", "[END]"]
        response = mychat(data["question"])
        for chunk in response:
            chunk_text = chunk.choices[0].delta.content
            print(chunk_text)
            if chunk_text is None:
                continue
            await websocket.send_text(chunk_text)
        await websocket.send_text("[END]")

        # await websocket.send_text("안녕하세요")


    except Exception as e:
        print(f"WebSocket 에러 발생: {e}")
    finally:
        await websocket.close()
        print("WebSocket 연결 종료")