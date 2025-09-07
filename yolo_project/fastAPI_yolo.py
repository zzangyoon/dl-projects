from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import uuid

SAVE_DIR = "images/"

class response(BaseModel):
    name : str
    x1 : float
    y1 : float
    x2 : float
    y2 : float
    conf : float
    cls : int

# Server
app = FastAPI(title="YOLO Inference")

# Model
model = YOLO("yolo11n.pt")


@app.post("/yolo", response_model = list[response])
async def predict(files: list[UploadFile] = File(...)):
    image_list = []
    for file in files:
        image = Image.open(file.file)
        ext = file.filename.split('.')[-1]
        file_name = f"{uuid.uuid4().hex[:9]}.{ext}"
        save_path = SAVE_DIR + file_name
        print("save_path ::: ", save_path)
        image.save(save_path)
        image_list.append(save_path)
    print("image_list ::: ", image_list)
    
    results = model(image_list, save=True, project="myresult")
    print(len(results))

    result_list = []
    for result in results:
        name = result.names
        boxes = result.boxes
        data = boxes.data

        for x1, y1, x2, y2, conf, cls in data:
            # print(i, " ::: " ,x1, y1, x2, y2, conf, cls, " ::: ", name[int(cls)])
            result_list.append(response(name=name[int(cls)], x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, cls=cls))

    return result_list