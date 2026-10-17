"""Filesystem layout. Data, artifacts and logs live under the project root, which can be
moved with the ``CLAUSEGUARD_HOME`` environment variable (tests use a temporary directory).
Configs come from the code checkout unless ``CLAUSEGUARD_CONFIGS`` points elsewhere."""

from __future__ import annotations

import os
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[2]


def root() -> Path:
    return Path(os.environ.get("CLAUSEGUARD_HOME", _CODE_ROOT))


def configs_dir() -> Path:
    """The code checkout's configs/, or ``CLAUSEGUARD_CONFIGS`` (used by tests)."""
    return Path(os.environ.get("CLAUSEGUARD_CONFIGS", _CODE_ROOT / "configs"))


def raw_dir(dataset: str) -> Path:
    return root() / "data" / "raw" / dataset


def processed_dir(module: str) -> Path:
    return root() / "data" / "processed" / module


def units_path(module: str) -> Path:
    return processed_dir(module) / "units.parquet"


def splits_dir() -> Path:
    return root() / "data" / "splits"


def data_cards_dir() -> Path:
    return root() / "data" / "data_cards"


def artifacts_dir(module: str, *, smoke: bool = False) -> Path:
    return root() / "artifacts" / ("smoke" if smoke else "runs") / module


def experiments_log() -> Path:
    return root() / "experiments" / "log.csv"


def reports_dir() -> Path:
    return root() / "reports" / "generated"
