"""Run a trained checkpoint on a single image and save a visualization.

Run: uv run python scripts/predict.py --config configs/yolov1.yaml \
        --ckpt runs/yolov1/best.pt --image path/to/img.jpg --out pred.png
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect objects in one image")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="prediction.png")
    parser.add_argument("--score-thresh", type=float, default=None)
    args = parser.parse_args()

    # TODO:
    #   1. cfg = load_config; device = get_device; enable_mps_fallback().
    #   2. model = build_model(cfg); load_checkpoint(args.ckpt, model); model.to(device).eval().
    #   3. Load image (PIL); apply the SAME val transforms as training; remember orig size.
    #   4. det = engine.inference.detect(model, image, device, score_thresh, nms_thresh).
    #   5. Map class ids -> names (cfg.data.classes); utils.viz.draw_boxes(...).
    #   6. Save annotated image to args.out.
    raise NotImplementedError("Implement scripts/predict.py")


if __name__ == "__main__":
    main()
