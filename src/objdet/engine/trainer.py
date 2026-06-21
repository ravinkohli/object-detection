"""Generic training loop, shared by all six detectors.

Design goal: the *model* returns its own loss dict in train mode, so this loop
stays detector-agnostic. Each model's ``forward(images, targets)`` should return
``(outputs, loss_dict)`` during training, where ``loss_dict`` maps name -> scalar
tensor (e.g. {"loss_cls": ..., "loss_box": ...}). The trainer just sums them,
backprops, steps, and logs.

This is plumbing — but YOU write it, so you understand exactly what happens each
step (the gradient flow, the device moves, the MLflow calls).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


class Trainer:
    """Owns the optimizer, scheduler, device, and the epoch loop."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Any | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        # TODO: store args; move model to device; stash grad-clip / log-interval from cfg.
        raise NotImplementedError("Implement Trainer.__init__")

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """Run one epoch; return averaged loss components.

        TODO (per batch):
            1. model.train(); move images + targets to self.device.
            2. loss_dict = model(images, targets)   # model computes its own loss
            3. loss = sum(loss_dict.values())
            4. optimizer.zero_grad(); loss.backward()
            5. (optional) clip gradients (torch.nn.utils.clip_grad_norm_).
            6. optimizer.step(); scheduler.step() if per-iter.
            7. every log_interval iters: mlflow_utils.log_metrics({...}, step=global_step).
            8. accumulate losses; return the epoch averages.

        Watch out: the collate_fn yields a *list* of targets (variable #boxes per
        image) — move each one to device individually.
        """
        raise NotImplementedError("Implement train_one_epoch")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None, epochs: int):
        """Full training: loop epochs, eval periodically, checkpoint best mAP.

        TODO:
            1. setup MLflow run + log_params(cfg) once (or do this in scripts/train.py).
            2. for epoch in range(epochs): train_one_epoch(...).
            3. if val_loader and epoch % eval_every == 0: run evaluator -> mAP;
               log_metrics({"mAP": ...}, step=epoch); save best checkpoint.
            4. step epoch-level scheduler if used.
            5. save final + best checkpoints (log_artifact them).
        """
        raise NotImplementedError("Implement Trainer.fit")


def save_checkpoint(path: str, model: torch.nn.Module, optimizer, epoch: int, extra: dict | None = None) -> None:
    """Save model+optimizer state to ``path`` (torch.save a dict). TODO."""
    raise NotImplementedError("Implement save_checkpoint")


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, map_location="cpu") -> dict:
    """Load a checkpoint into ``model`` (and optimizer). Return the loaded dict. TODO."""
    raise NotImplementedError("Implement load_checkpoint")
