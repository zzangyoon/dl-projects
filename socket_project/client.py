# (언리얼이 FastAPI 서버에 데이터를 보낸다고 가정)
import websockets
import asyncio
import json

WEBSOCKET_URL = "ws://localhost:8000/ws/streaming"

async def send_message(question):
    async with websockets.connect(WEBSOCKET_URL) as websocket:
        # FastAPI에게 메시지 전송
        # json.dumps 는 dict 형태를 str로 바꿔줌
        json_data = json.dumps(
            {"question" : question},
            ensure_ascii=False
        )
        await websocket.send(json_data)

        # FastAPI 서버에서 응답 받기
        while True:
            token = await websocket.recv()
            if token == "[END]":
                break

            yield token

async def main():
    question = "안녕하세요"
    async for token in send_message(question):
        print(token, end="", flush=True)

# 실행하기
asyncio.run(main())