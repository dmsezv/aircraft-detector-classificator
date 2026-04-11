# src/utils/labels.py
import json
import logging

from src.utils.env_config import settings

logger = logging.getLogger(__name__)


def load_id2label() -> dict:
    try:
        with open(settings.classifier_config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            labels = data.get("id2label", {})
            logger.info(f"Aircraft classes: {len(labels)}")
            return labels
    except FileNotFoundError:
        logger.error(f"No config file here: {settings.classifier_config_path}")
        return {}
    except Exception as e:
        logger.error(f"{e}")
        return {}


AIRPLANE_CLASSES = load_id2label()
