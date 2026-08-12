"""Config-coherence tests.

The convention is that no numeric lives in code, which only pays off if the configs
agree with each other. These catch the failure mode where two files drift apart —
the priority weights no longer sum to 1, the gate demands more precision than the
threshold is fitted to hold, or the drift monitor watches a dropped feature.
"""
import yaml

from src.config import CONFIG_DIR, SLOT_ROOT, HAS_INFRA


def test_all_three_configs_load(cfg):
    assert set(cfg) == {"model", "feature", "serving"}
    assert all(isinstance(v, dict) and v for v in cfg.values())


def test_config_dir_holds_exactly_the_expected_files():
    assert sorted(p.name for p in CONFIG_DIR.glob("*.yaml")) == [
        "feature_config.yaml", "model_config.yaml", "serving_config.yaml"]


def test_every_config_opens_with_a_header_comment():
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert yaml.safe_load(text), f"{path.name} is empty or invalid"
        assert text.lstrip().startswith("#"), f"{path.name} has no header comment"


def test_priority_weights_sum_to_one(cfg):
    w = cfg["model"]["priority"]["weights"]
    assert abs(w["probability"] + w["volume"] - 1.0) < 1e-9, w


def test_split_fraction_is_sane(cfg):
    assert 0.0 < cfg["model"]["split"]["test_frac"] < 1.0


def test_precision_gate_cannot_exceed_the_fitted_floor(cfg):
    """A gate stricter than what the threshold is fitted to hold can never pass."""
    fitted = cfg["model"]["operating_point"]["min_precision"]
    gate = cfg["model"]["acceptance"]["precision_min"]
    assert gate <= fitted, f"gate {gate} exceeds fitted floor {fitted}"


def test_decile_counts_agree_across_configs(cfg):
    """The API tiers deciles; the model builds them. A mismatch mis-tiers alerts."""
    assert cfg["model"]["priority"]["n_deciles"] == cfg["serving"]["decile_tiering"]["n_deciles"]


def test_tiering_actions_are_within_range(cfg):
    tier = cfg["serving"]["decile_tiering"]
    n = tier["n_deciles"]
    assert all(1 <= int(k) <= n for k in tier["actions"])
    assert set(tier["alert_deciles"]) <= set(range(1, n + 1))


def test_gain_floor_below_gain_cap(cfg):
    bal = cfg["feature"]["screen"]["balance"]
    assert bal["gain_floor_pct"] < bal["gain_cap_pct"]
    assert bal["min_selected"] <= bal["min_pool_size"]


def test_screen_thresholds_are_probabilities(cfg):
    scr = cfg["feature"]["screen"]
    assert 0.5 < scr["leak"]["single_feature_auc_min"] <= 1.0
    assert 0.0 < scr["redundancy"]["abs_corr_max"] <= 1.0
    assert 0.0 < scr["near_constant"]["max_zero_frac"] <= 1.0


def test_dataset_filenames_have_the_right_extensions(cfg):
    data = cfg["model"]["data"]
    assert data["dataset"].endswith(".parquet")
    assert data["manifest"].endswith(".json")
    assert data["selected"].endswith(".json")
    assert data["screen_audit"].endswith(".csv")


def test_analyst_cost_is_configured(cfg):
    """The workload gate is meaningless without a per-alert cost."""
    assert cfg["serving"]["scoring"]["analyst_minutes_per_alert"] > 0


def test_drift_monitor_watches_features_the_model_has(selected):
    """A drift config pointing at dropped columns monitors nothing."""
    with open(SLOT_ROOT / "mlops" / "drift_config.yaml", encoding="utf-8") as fh:
        drift = yaml.safe_load(fh)
    watched = set(drift["feature_drift"]["top_features"])
    missing = watched - set(selected["features"])
    assert not missing, f"drift_config watches features the model does not use: {missing}"


def test_drift_gates_match_the_acceptance_gates(cfg):
    """Monitor and promotion gate must enforce the same FN ceiling and floor."""
    with open(SLOT_ROOT / "mlops" / "drift_config.yaml", encoding="utf-8") as fh:
        drift = yaml.safe_load(fh)
    assert drift["performance"]["fn_rate"]["target_max"] == cfg["model"]["acceptance"]["fn_rate_max"]
    assert (drift["performance"]["precision_at_operating_point"]["floor"]
            == cfg["model"]["operating_point"]["min_precision"])


def test_standalone_mode_is_detected():
    """This tree ships without _infra/, so config must report that rather than raise."""
    assert isinstance(HAS_INFRA, bool)
