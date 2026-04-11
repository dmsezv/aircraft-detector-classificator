import logging
import warnings

import fiftyone as fo
import hydra
from omegaconf import DictConfig

warnings.filterwarnings("ignore", category=SyntaxWarning)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    target_class = cfg.dataset.target_class_name
    export_dir = cfg.dataset.processed_data_dir

    for split in cfg.dataset.splits:
        logger.info(f"Get data for {split}")

        dataset = fo.zoo.load_zoo_dataset(
            "coco-2017",
            split=split,
            label_types=["detections"],
            classes=[target_class],
            # max_samples=50,
        )

        dataset.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            split=split,
            classes=[target_class],
        )


if __name__ == "__main__":
    main()
