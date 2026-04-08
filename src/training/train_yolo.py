import logging
import os

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

from src.utils.utils import flatten_dict

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Strat train with params:\n{OmegaConf.to_yaml(cfg)}")

    db_path = cfg.mlflow.tracking_uri.replace("sqlite:///", "")
    db_dir = os.path.dirname(db_path)

    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow.experiment_name
    os.environ["MLFLOW_TRACKING_URI"] = cfg.mlflow.tracking_uri

    logger.info(f"Model: {cfg.model.weights}")
    model = YOLO(cfg.model.weights)

    dataset_yaml_path = os.path.join(
        cfg.core.work_dir, cfg.dataset.processed_data_dir, "dataset.yaml"
    )
    runs_dir = os.path.join(cfg.core.work_dir, "runs")

    with mlflow.start_run(run_name=cfg.model.name):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        flat_cfg = flatten_dict(cfg_dict)
        mlflow.log_params(flat_cfg)

        model.train(
            data=dataset_yaml_path,
            epochs=cfg.training.epochs,
            imgsz=cfg.training.imgsz,
            batch=cfg.training.batch_size,
            device=cfg.training.device,
            project=runs_dir,
            name=cfg.model.name,
            exist_ok=True,
            verbose=True,
        )

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
