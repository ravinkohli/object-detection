"""Evaluate a trained checkpoint and print COCO mAP.

Run: uv run python scripts/eval.py --config configs/faster_rcnn.yaml --ckpt runs/faster_rcnn/best.pt
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a detector checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    # TODO:
    #   1. cfg = load_config; device = get_device; enable_mps_fallback().
    #   2. Build val loader + coco_gt (reuse scripts.train.build_dataloaders).
    #   3. model = build_model(cfg); load_checkpoint(args.ckpt, model); model.to(device).
    #   4. metrics = engine.evaluator.evaluate(model, val_loader, device, coco_gt).
    #   5. print(metrics)  # mAP / AP50 / AP75.
    raise NotImplementedError("Implement scripts/eval.py")


if __name__ == "__main__":
    main()
