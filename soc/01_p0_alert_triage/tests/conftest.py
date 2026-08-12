"""Shared pytest fixtures for the Alert Triage (#01) model."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SLOT = Path(__file__).resolve().parents[1]
if str(SLOT) not in sys.path:
    sys.path.insert(0, str(SLOT))

from src.config import DATA_DIR, OUTPUT_DIR, load_all_configs  # noqa: E402

BUNDLE_PATH = OUTPUT_DIR / "lgbm_model.pkl"
METRICS_PATH = OUTPUT_DIR / "metrics.json"


@pytest.fixture(scope="session")
def cfg():
    return load_all_configs()


@pytest.fixture(scope="session")
def selected():
    """The screened feature set — committed, so always present."""
    path = DATA_DIR / "native_selected.json"
    if not path.exists():
        pytest.skip("native_selected.json missing — run src/pipeline.py")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def metrics():
    """Metrics of the last training run. Skips when the model hasn't been built."""
    if not METRICS_PATH.exists():
        pytest.skip("outputs/metrics.json missing — run src/pipeline.py")
    with open(METRICS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def bundle():
    if not BUNDLE_PATH.exists():
        pytest.skip("outputs/lgbm_model.pkl missing — run src/pipeline.py")
    import joblib
    return joblib.load(BUNDLE_PATH)


@pytest.fixture(scope="session")
def trained():
    """The trained LightGBMTriage plus its serving bundle."""
    if not BUNDLE_PATH.exists():
        pytest.skip("outputs/lgbm_model.pkl missing — run src/pipeline.py")
    from src.models.lightgbm_triage import LightGBMTriage
    return LightGBMTriage.load(str(BUNDLE_PATH))


@pytest.fixture(scope="session")
def client(bundle):
    """FastAPI test client. Depends on `bundle` so it skips before a first build."""
    pytest.importorskip("httpx", reason="httpx is required by fastapi.testclient")
    from fastapi.testclient import TestClient
    from src.serving.flow_triage_api import app
    return TestClient(app)


@pytest.fixture
def benign_flow(bundle):
    """A minimal request body: every model feature present, all zero."""
    return {f: 0 for f in bundle["features"]}


# ---- toy data, so the unit tests do not need the 300 MB parquet ---------------

@pytest.fixture
def toy_flows():
    """Two native features that separate cleanly, plus a label and a class name.

    Sized above `screen.ranker.min_child_samples` (200) so the real screening
    config can fit on it — below that LightGBM bins every feature away and the
    booster comes back with no features at all.
    """
    rng = np.random.RandomState(0)
    n, half = 4000, 2000
    y = np.r_[np.zeros(half, dtype=int), np.ones(half, dtype=int)]

    # Informative but NOT leak-grade. Perfectly separated features score AUC 1.0,
    # which the leak screen removes by design — the classes must overlap or the
    # screen correctly throws every feature away and there is nothing left to fit.
    benign_bytes = rng.normal(100, 80, half)
    attack_bytes = rng.normal(300, 80, half)           # ~2.5 sd apart -> AUC ~0.96

    benign_port = rng.randint(1024, 65535, half)
    # 60% of attacks hit a service port, 40% look like ordinary traffic -> AUC ~0.8
    attack_port = np.where(rng.random(half) < 0.6,
                           rng.choice([80, 22], half),
                           rng.randint(1024, 65535, half))

    return pd.DataFrame({
        "L4_DST_PORT": np.r_[benign_port, attack_port],
        "SRC_TO_DST_SECOND_BYTES": np.r_[benign_bytes, attack_bytes],
        "DEAD_COLUMN": np.zeros(n),                    # near-constant NOISE
        "y_is_attack": y,
        "attack_class": np.where(y == 1, "DoS attacks-Hulk", "BENIGN"),
    })
