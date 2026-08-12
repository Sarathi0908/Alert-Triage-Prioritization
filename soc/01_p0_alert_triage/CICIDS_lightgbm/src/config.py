"""
config.py  —  config loading + shared-platform (_infra) access for SOC #01
              (CSE-CIC-IDS2018 / CICIDS2017 LightGBM variant).

Keeps the rest of src/ free of path juggling: it resolves the model-slot root
and the repo root, loads the YAML configs, and imports `_infra/` utilities by
file path (so models genuinely reuse the shared platform without needing
`_infra` to be an installed package).

Difference from the in-repo slot: this copy is a STANDALONE tree, so `_infra/`
may not exist above it. `_find_repo_root` therefore falls back to the slot root
instead of raising at import time, and `infra()` raises only when a shared-
platform module is actually requested.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import yaml

# This file lives at <slot>/src/config.py -> slot root is two parents up.
SLOT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = SLOT_ROOT / "configs"

DATA_DIR = SLOT_ROOT / "data"
OUTPUT_DIR = SLOT_ROOT / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"


def _find_repo_root(start: Path) -> Path | None:
    """Walk up until a directory that contains `_infra/`; None when standalone."""
    for p in [start, *start.parents]:
        if (p / "_infra").is_dir():
            return p
    return None


REPO_ROOT = _find_repo_root(SLOT_ROOT)
HAS_INFRA = REPO_ROOT is not None


def load_config(name: str) -> dict[str, Any]:
    """Load one of the slot's YAML configs by stem, e.g. load_config('model')."""
    path = CONFIG_DIR / f"{name}_config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_all_configs() -> dict[str, dict]:
    return {n: load_config(n) for n in ("model", "feature", "serving")}


def infra(rel_module_path: str):
    """Import a module from the shared platform by its path under `_infra/`.

    Example:
        psi = infra("mlops/monitoring_drift/drift_detection/psi_monitor.py")
    """
    if REPO_ROOT is None:
        raise ImportError(
            f"cannot load _infra module {rel_module_path!r}: no `_infra/` directory "
            f"above {SLOT_ROOT} — this slot is running standalone. Place it under a "
            f"checkout that contains `_infra/` to use shared-platform utilities."
        )
    file_path = REPO_ROOT / "_infra" / rel_module_path
    mod_name = "_infra_" + rel_module_path.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load _infra module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_path(rel: str) -> str:
    """Resolve a slot-relative path (e.g. a data path from config) to absolute."""
    p = Path(rel)
    return str(p if p.is_absolute() else (SLOT_ROOT / p))
