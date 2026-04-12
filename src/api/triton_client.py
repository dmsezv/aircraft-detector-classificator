import numpy as np
import tritonclient.http as httpclient

from src.utils.env_config import settings


class TritonClient:
    def __init__(self, url=settings.triton_url):
        self.client = httpclient.InferenceServerClient(url=url)

    def infer_detector(self, input_tensor: np.ndarray) -> np.ndarray:
        inputs = [httpclient.InferInput("images", input_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_tensor)

        outputs = [httpclient.InferRequestedOutput("output0")]
        response = self.client.infer("yolov8_airplane", inputs, outputs=outputs)

        return response.as_numpy("output0")

    def infer_classifier(self, input_tensor: np.ndarray) -> np.ndarray:
        inputs = [httpclient.InferInput("pixel_values", input_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_tensor)

        outputs = [httpclient.InferRequestedOutput("logits")]
        response = self.client.infer("convnext_classifier", inputs, outputs=outputs)

        return response.as_numpy("logits")


triton_client = TritonClient()
