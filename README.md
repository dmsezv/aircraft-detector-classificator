# Aircraft Detector & Classificator

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Triton](https://img.shields.io/badge/Triton_Inference_Server-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/triton-inference-server)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org/)

**Протестировать сервис вживую:** [http://195.209.218.220:8501](http://195.209.218.220:8501)

Видео превью:

<picture>
    <img alt="" src="https://github.com/dmsezv/aircraft-detector-classificator/blob/main/pics/example.gif">
</picture>

---

## Архитектура проекта

Проект построен на базе микросервисной архитектуры и реализует Two-Stage Inference пайплайн:

1. **Детекция:** Поиск самолета на изображении и выделение Bounding Box.
2. **Классификация:** Определение модели самолета.

### Система состоит из трех изолированных контейнеров:

- **Frontend:** Интерактивный веб-интерфейс на `Streamlit` с использованием `OpenCV` для отрисовки рамок и результатов классификации.
- **Backend:** REST API на базе `FastAPI`.
- **Inference Server:** NVIDIA Triton Inference Server

---

## Обучение и анализ моделей детекции

В качестве базовой модели детекции взята архитектура **YOLOv8n**. В ходе исследования было проведено обучение и сравнение трех структурных модификаций сети на датасете с одним классом. Для логирования метрик и отслеживания прогресса обучения использовался **MLflow**.

- [Аналитический отчет по моделям](./ссылка_на_отчет.pdf)
- [Модификации архитектуры](./ссылка_на_папку_с_моделями)

---

## Деплой и запуск (Docker)

Шаги для запуска

1.  Клонирование репозитория

    ```
    git clone https://github.com/dmsezv/aircraft-detector-classificator.git

    cd aircraft-detector-classificator
    ```

2.  Загрузка весов моделей
  
    > Поскольку файлы весов (.onnx, .pt) исключены из системы контроля версий, их необходимо перенести в директорию model_repository вручную.

        Структура model_repository должна выглядеть следующим образом:

        ```
        model_repository/
        ├── convnext_classifier/
        │   ├── 1/
        │   │   ├── config.json
        │   │   └── model.onnx
        │   └── config.pbtxt
        └── yolov8_airplane/
            ├── 1/
            │   └── model.onnx
            └── config.pbtxt
        ```

4.  Сборка и запуск контейнеров

    ```
    docker compose up -d --build
    ```

5.  Доступ к сервисам

    ```
    Frontend (UI): http://<IP_СЕРВЕРА>:8501

    Backend (Swagger UI): http://<IP_СЕРВЕРА>:8080/docs
    ```
