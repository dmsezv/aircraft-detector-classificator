import logging
import os
import shutil

from ultralytics import YOLO

from src.utils.env_config import settings

logger = logging.getLogger(__name__)


def export_detector_to_triton():
    model = YOLO(settings.detector_path)
    exported_path = model.export(format="onnx", imgsz=640, opset=18)

    os.makedirs(settings.triton_model_dir, exist_ok=True)

    destination = os.path.join(settings.triton_model_dir, "model.onnx")
    shutil.move(exported_path, destination)

    logger.info(f"ONNX saved path: {destination}")


if __name__ == "__main__":
    export_detector_to_triton()
