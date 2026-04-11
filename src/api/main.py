import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

from src.api.triton_client import triton_client
from src.utils.aircrafts_lables import AIRPLANE_CLASSES

app = FastAPI(title="Aircraft Detector API")


@app.post("/predict_airplane")
async def predict_airplane(file: UploadFile = File(...)):
    image_bytes = await file.read()

    np_array = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    img = LetterBox(640, auto=False, stride=32)(image=img_cv)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    raw_output = triton_client.infer_detector(img)
    preds = torch.from_numpy(raw_output)
    results = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)

    detections = []
    if len(results) > 0 and len(results[0]) > 0:
        det = results[0]
        det[:, :4] = scale_boxes((640, 640), det[:, :4], img_cv.shape[:2]).round()

        for *xyxy, conf_det, _ in det:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            crop = img_cv[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (224, 224))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

            crop_float = crop_rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            crop_norm = (crop_float - mean) / std

            crop_input = crop_norm.transpose(2, 0, 1)
            crop_input = np.expand_dims(crop_input, axis=0)

            logits = triton_client.infer_classifier(crop_input)

            e_x = np.exp(logits[0] - np.max(logits[0]))
            probabilities = e_x / e_x.sum()
            conf_cls = float(np.max(probabilities))

            class_id = np.argmax(logits[0])
            model_name = AIRPLANE_CLASSES.get(str(class_id), f"Unknown_{class_id}")

            detections.append(
                {
                    "confidence_detection": round(float(conf_det), 3),
                    "confidence_classification": round(float(conf_cls), 3),
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int(class_id),
                    "label": model_name,
                }
            )

    return {"filename": file.filename, "total": len(detections), "detections": detections}
