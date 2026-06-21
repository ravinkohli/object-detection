"""Config loading: YAML file -> plain Python dict (or your own dataclass).

The configs in ``configs/*.yaml`` are the single source of truth for an
experiment (dataset subset, model hyperparameters, optimizer, MLflow run name,
etc.). The training/eval scripts load one of them and pass it around.

Keep this dead simple. You can return a nested dict, or (nicer) parse it into a
dataclass so you get attribute access + validation. Your call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a dict.

    Args:
        path: path to a file under ``configs/``.

    Returns:
        Nested dict mirroring the YAML structure.

    TODO:
        1. Open ``path`` and parse with ``yaml.safe_load``.
        2. (Optional) merge with a ``defaults`` section or a base config so the
           per-model YAMLs only override what differs.
        3. (Optional) validate required keys are present and fail loudly if not.
        4. Return the dict (or convert to a dataclass — see ``Config`` below).
    """
    raise NotImplementedError("Implement load_config")


# (Optional, recommended) Turn the dict into a typed object for attribute access.
# Sketch only — flesh out the fields to match your YAML schema.
#
# from dataclasses import dataclass
#
# @dataclass
# class Config:
#     model: dict
#     data: dict
#     optim: dict
#     mlflow: dict
#     ...
#
#     @classmethod
#     def from_yaml(cls, path: str | Path) -> "Config":
#         ...
