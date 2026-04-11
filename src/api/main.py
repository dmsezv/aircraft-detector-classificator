from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io


app = FastAPI(title="Aircraft Detector API")

MODEL_PATH = "runs/yolov8n_baseline/weights/best.pt" 
model = YOLO(MODEL_PATH)

@app.post("/predict_airplane")
async def predict_airplane(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))
    results = model(image)

    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # [x_min, y_min, x_max, y_max]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            
            detections.append({
                "confidence": conf,
                "bbox": [
                    round(x1), 
                    round(y1), 
                    round(x2), 
                    round(y2)
                ]
            })

    return {
        "filename": file.filename,
        "total": len(detections),
        "detections": detections
    }