from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    detector_path: str = ""
    triton_model_dir: str = ""
    classifier_config_path: str = "config.json"
    tritin_url: str = ""
    api_url: str = "http://localhost:8000/predict_airplane"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
