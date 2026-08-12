"""Shared pytest fixtures for the Alert Triage (#01) CSE-CIC-IDS2018 model."""
import json
import sys
from pathlib import Path

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
    """The screened feature set (data/native_selected.json) — committed, always present."""
    path = DATA_DIR / "native_selected.json"
    if not path.exists():
        pytest.skip("native_selected.json missing — run src/features/feature_screener.py")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def metrics():
    """Metrics of the last training run. Skips when the model hasn't been built."""
    if not METRICS_PATH.exists():
        pytest.skip("outputs/metrics.json missing — run src/models/lightgbm_triage.py")
    with open(METRICS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def bundle():
    """The trained bundle. Skips when the model hasn't been built."""
    if not BUNDLE_PATH.exists():
        pytest.skip("outputs/lgbm_model.pkl missing — run src/models/lightgbm_triage.py")
    import joblib
    return joblib.load(BUNDLE_PATH)


@pytest.fixture(scope="session")
def client(bundle):
    """FastAPI test client. Depends on `bundle` so it skips before a first build."""
    pytest.importorskip("httpx", reason="httpx is required by fastapi.testclient")
    from fastapi.testclient import TestClient
    from src.serving.api import app
    return TestClient(app)


@pytest.fixture
def benign_flow(bundle):
    """A minimal request body: every model feature present, all zero."""
    return {f: 0 for f in bundle["features"]}
