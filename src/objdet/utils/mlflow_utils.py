"""Thin helpers around MLflow so the engine code stays readable.

MLflow concepts you'll use:
  - experiment : a named bucket of runs (use one per model family, e.g. "faster_rcnn").
  - run        : one training execution; has params, metrics, artifacts, tags.
  - params     : hyperparameters logged once (from the config).
  - metrics    : numbers logged over time with a ``step`` (losses per iter, mAP per epoch).
  - artifacts  : files (checkpoints, sample prediction images, the config copy).

Tracking is local by default: runs are written to ``./mlruns``. View them with
``uv run mlflow ui`` and open http://localhost:5000.

Docs / API: https://mlflow.org/docs/latest/python_api/mlflow.html
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


def setup_mlflow(experiment_name: str, tracking_uri: str | None = None) -> None:
    """Point MLflow at a tracking store and select/create the experiment.

    TODO:
        1. If ``tracking_uri`` given, ``mlflow.set_tracking_uri(tracking_uri)``
           (default file store is ./mlruns, which is fine for local).
        2. ``mlflow.set_experiment(experiment_name)``.
    """
    raise NotImplementedError("Implement setup_mlflow")


@contextmanager
def start_run(run_name: str | None = None, tags: dict[str, str] | None = None):
    """Context manager wrapping ``mlflow.start_run``.

    Usage:
        with start_run("yolov1-subset"):
            log_params(cfg)
            ... training loop ...

    TODO:
        1. ``with mlflow.start_run(run_name=run_name, tags=tags) as run:``
        2. ``yield run``.
    """
    raise NotImplementedError("Implement start_run")


def log_params(params: dict[str, Any]) -> None:
    """Log hyperparameters (flatten nested dicts; MLflow params are flat str->str).

    TODO: flatten ``params`` (e.g. "optim.lr" -> 0.001) then ``mlflow.log_params``.
    """
    raise NotImplementedError("Implement log_params")


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log one or more scalar metrics at an optional ``step``.

    TODO: ``mlflow.log_metrics(metrics, step=step)``.
    """
    raise NotImplementedError("Implement log_metrics")


def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    """Upload a file (checkpoint, image, config) to the run's artifact store.

    TODO: ``mlflow.log_artifact(local_path, artifact_path)``.
    """
    raise NotImplementedError("Implement log_artifact")
