"""Train any of the six detectors from a config file.

Run: uv run python scripts/train.py --config configs/faster_rcnn.yaml

This is the top-level wiring: config -> data -> model -> optimizer -> Trainer.fit,
with MLflow tracking. It stays model-agnostic by dispatching on cfg['model']['name'].
"""

from __future__ import annotations

import argparse


def build_model(cfg):
    """Dispatch cfg['model']['name'] -> the right nn.Module.

    TODO: build backbone (models.backbone.build_backbone) then one of
    FastRCNN / FasterRCNN / YOLOv1 / DETR / DINO / RTDETR with num_classes from cfg.
    """
    raise NotImplementedError("Implement build_model")


def build_dataloaders(cfg):
    """Build train/val DataLoaders from the subset + transforms.

    TODO:
        1. Load COCO; select subset ids (data/subset); train/val split.
        2. Build transforms (data/transforms.build_transforms) for train & val.
        3. CocoDetectionDataset(...) x2 with a SHARED cat_id_map.
        4. DataLoader(..., collate_fn=data.coco.collate_fn). Return (train, val, coco_gt).
    """
    raise NotImplementedError("Implement build_dataloaders")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = parser.parse_args()

    # TODO:
    #   1. cfg = load_config(args.config); set_seed; device = get_device(cfg.device).
    #   2. enable_mps_fallback().
    #   3. train_loader, val_loader, coco_gt = build_dataloaders(cfg).
    #   4. model = build_model(cfg).to(device).
    #   5. optimizer (SGD for R-CNN/YOLO, AdamW + lower backbone lr for DETR family).
    #   6. scheduler (MultiStepLR from cfg.optim.lr_steps).
    #   7. setup_mlflow(cfg.mlflow.experiment); with start_run(cfg.mlflow.run_name):
    #          log_params(cfg)
    #          Trainer(model, optimizer, device, scheduler, cfg).fit(train, val, epochs)
    raise NotImplementedError("Implement scripts/train.py")


if __name__ == "__main__":
    main()
