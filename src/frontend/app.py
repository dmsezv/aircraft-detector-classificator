import os
import sys

import numpy as np
import cv2
import requests
import streamlit as st
from PIL import Image

from src.utils.env_config import settings


st.set_page_config(page_title="Aircraft Detector", layout="wide")

st.title("Детекция и классификация самолетов")

uploaded_file = st.file_uploader("Выберите фото самолета", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)

    image_placeholder = st.empty()
    image_placeholder.image(original_image, use_container_width=True)

    if st.button("Найти самолеты", type="primary"):
        with st.spinner("Обработка изображения"):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                response = requests.post(settings.api_url, files=files)

                if response.status_code == 200:
                    data = response.json()
                    detections = data.get("detections", [])

                    if not detections:
                        st.warning("Самолеты не найдены на этой картинке.")
                    else:
                        st.success(f"Найдено самолетов: {data['total']}")

                        result_image_np = np.array(original_image.convert("RGB"))

                        bbox_color = (57, 255, 20)
                        text_color = (0, 0, 0)
                
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        font_width = 1
                        bbox_width = 3

                        for det in detections:
                            bbox = det["bbox"]
                            label = det["label"]
                            conf_det = det["confidence_detection"]
                            conf_cls = det["confidence_classification"]

                            x1, y1, x2, y2 = bbox
                            
                            cv2.rectangle(
                                result_image_np, 
                                (x1, y1), 
                                (x2, y2), 
                                bbox_color, 
                                bbox_width
                            )

                            text = f"{label} (Det: {conf_det:.2f}% Cls: {conf_cls:.2f}%)"

                            (text_width, text_height), baseline = cv2.getTextSize(
                                text, font, font_scale, font_width
                            )

                            cv2.rectangle(
                                result_image_np,
                                (x1 - 2, y1 - text_height - baseline - 5),
                                (x1 + text_width, y1),
                                bbox_color,
                                -1
                            )

                            cv2.putText(
                                result_image_np,
                                text,
                                (x1, y1 - baseline - 2),
                                font,
                                font_scale,
                                text_color,
                                font_width,
                                cv2.LINE_AA 
                            )

                        image_placeholder.image(result_image_np, use_container_width=True)
                        
                        with st.expander("Response JSON"):
                            st.json(data)

                else:
                    st.error(f"Ошибка сервера: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Не удалось подключиться к API: {e}")
