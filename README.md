# object-detection — the evolution of object detection, from scratch

A learning project: implement the major object-detection architectures **by hand in
PyTorch** to understand how the field evolved, track experiments with **MLflow**, and
train on a small **COCO** subset.

> **This repo is a scaffold of guided stubs.** Almost every function/class raises
> `NotImplementedError` and carries an inline docstring with the concept, expected
> tensor shapes, an ordered TODO list, the library calls to use, and the relevant
> paper section. **You write the bodies.** The plumbing (structure, configs, env) is
> set up for you.

## The six detectors (three eras)

| Era | Model | Key idea | Files |
|-----|-------|----------|-------|
| Two-stage | **Fast R-CNN** | shared conv + ROI pooling over *external* (Selective Search) proposals | `models/fast_rcnn.py`, `data/proposals.py` |
| Two-stage | **Faster R-CNN** | learns proposals with an **RPN** (replaces Selective Search) | `models/rpn.py`, `models/faster_rcnn.py` |
| One-stage | **YOLOv1** | one regression from pixels to an `S×S×(B*5+C)` grid; no proposals/anchors | `models/yolov1.py`, `losses/yolo_loss.py` |
| Transformer | **DETR** | set prediction with a transformer + **Hungarian matching**; no NMS | `models/detr.py`, `ops/hungarian_matcher.py`, `losses/set_criterion.py` |
| Transformer | **DINO** | Deformable DETR + **contrastive denoising** + query selection | `models/dino.py`, `ops/deformable_attn.py` |
| Transformer | **RT-DETR** | real-time: **efficient hybrid encoder** (AIFI + CCFM) + IoU-aware queries | `models/rtdetr.py` |

## Setup (done)

Environment is managed with **uv** (Python 3.12). It's already created:

```bash
uv sync --extra dev                      # (re)install deps incl. pytest
uv run python -c "import torch; print(torch.backends.mps.is_available())"   # -> True
```

This machine is an **Apple M5 Pro (MPS, no CUDA)**. `utils/device.py` selects MPS and
sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so any op missing on MPS falls back to CPU.

## What's hand-written vs. borrowed (the "lean/pragmatic" boundary)

- **You implement:** box delta encode/decode, anchors, matcher/sampler, RPN, ROI head,
  transformer encoder/decoder, positional encodings, object queries, deformable
  attention, the Hungarian matcher's cost, the set-prediction criterion, **all six
  detectors**, and **all losses**.
- **Borrowed (already wired in the stub TODOs):** `torchvision.ops` primitives
  (`nms`, `roi_align`, `box_iou`/`generalized_box_iou`, `box_convert`, clip),
  `torchvision.models` backbones (enables the ImageNet `pretrained_backbone` toggle),
  `pycocotools` for COCO loading + mAP, and `scipy` for `linear_sum_assignment`.
- **From scratch:** backbones default to **random init**; flip `pretrained_backbone:
  true` in a config to warm-start (DETR-family really wants this — they're slow).

## Data: small COCO subset

The default downloads only **val2017 (~1GB) + annotations** and carves train/val
subsets from it (classes + image count set in each config under `data:`), so the whole
loop runs locally. Use `--full` to also pull `train2017` (18GB) later.

```bash
uv run python scripts/download_coco.py            # val2017 + annotations
uv run python scripts/download_coco.py --full     # also train2017 (big)
```

## Experiment tracking: MLflow

Runs are logged to a local `./mlruns` store. After (or during) training:

```bash
uv run mlflow ui        # open http://localhost:5000
```

`utils/mlflow_utils.py` has the helpers; `engine/trainer.py` shows where to log params
(the config), metrics (losses per step, mAP per epoch), and artifacts (checkpoints,
sample prediction images).

## Suggested build order

Work **bottom-up** — each step is exercised by the next. Run `uv run pytest` as you go.

1. **`ops/boxes.py`** → `ops/anchors.py` → `ops/matcher.py` — the foundation. Add tests
   in `tests/test_boxes.py` (un-skip them).
2. **`data/`** — `coco.py`, `subset.py`, `transforms.py` + `scripts/download_coco.py`.
3. **`models/backbone.py`**, then **Fast R-CNN** (`models/fast_rcnn.py`,
   `losses/detection_loss.py`, `data/proposals.py` + `scripts/precompute_proposals.py`).
4. **`models/rpn.py`** → **Faster R-CNN** (`models/faster_rcnn.py` — reuses the ROI head).
5. **YOLOv1** (`models/backbone.py:YOLOBackbone`, `models/yolov1.py`, `losses/yolo_loss.py`).
6. **DETR** (`models/transformer.py` → `models/detr.py`, `ops/hungarian_matcher.py`,
   `losses/set_criterion.py`) — the cleanest transformer detector; do it first of the three.
7. **DINO** (`ops/deformable_attn.py` → `models/dino.py`; start as Deformable DETR, then
   add query selection, then contrastive denoising).
8. **RT-DETR** (`models/rtdetr.py` — reuse DINO's deformable decoder; add AIFI + CCFM).
9. **`engine/`** (`trainer.py`, `evaluator.py`, `inference.py`) + `scripts/{train,eval,predict}.py`.

## Running things (once implemented)

```bash
uv run python scripts/train.py   --config configs/faster_rcnn.yaml
uv run python scripts/eval.py    --config configs/faster_rcnn.yaml --ckpt runs/faster_rcnn/best.pt
uv run python scripts/predict.py --config configs/faster_rcnn.yaml --ckpt runs/faster_rcnn/best.pt --image some.jpg
```

## Layout

```
configs/            one YAML per detector (filled templates; the experiment's source of truth)
src/objdet/
  data/             COCO dataset, subset builder, transforms, Selective Search proposals
  ops/              box math, anchors, matcher, Hungarian matcher, deformable attention
  models/           backbones + the 6 detectors + shared transformer blocks
  losses/           R-CNN multitask loss, YOLO loss, DETR-family set criterion
  engine/           trainer, COCO evaluator, inference/post-processing
  utils/            config, device, mlflow, viz
scripts/            download_coco, precompute_proposals, train, eval, predict
tests/              test_boxes, test_smoke (skipped until you implement + un-skip)
```

## Notes

- Training from scratch on a subset is for *seeing the mechanics work and loss/mAP move*,
  not paper numbers. The `pretrained_backbone` toggle and `--full` data flag are how you
  push toward real accuracy later.
- Loading **full** pretrained DETR/DINO/RT-DETR weights would require matching the
  reference architecture's state-dict keys (or going via HuggingFace `transformers`) —
  noted in those model stubs; the default path is training your own.
