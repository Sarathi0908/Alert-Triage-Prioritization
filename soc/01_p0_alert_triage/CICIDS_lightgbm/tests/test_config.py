"""Config-coherence tests.

The repo convention is that no numeric lives in code, which only pays off if the
configs are internally consistent. These tests catch the failure mode where two
files drift apart — e.g. the priority weights no longer sum to 1, or the drift
monitor watches a feature the model no longer has.
"""
import yaml

from src.config import CONFIG_DIR, SLOT_ROOT


def test_all_three_configs_load(cfg):
    assert set(cfg) == {"model", "feature", "serving"}
    assert all(isinstance(v, dict) and v for v in cfg.values())


def test_priority_weights_sum_to_one(cfg):
    w = cfg["model"]["priority"]["weights"]
    assert abs(w["probability"] + w["volume"] - 1.0) < 1e-9, (
        f"priority weights must sum to 1, got {w}")


def test_split_fraction_is_sane(cfg):
    frac = cfg["model"]["split"]["test_frac"]
    assert 0.0 < frac < 1.0


def test_precision_floor_and_gate_agree(cfg):
    """The gate must not demand more precision than the threshold is fitted to hold."""
    fitted = cfg["model"]["operating_point"]["min_precision"]
    gate = cfg["model"]["acceptance"]["precision_min"]
    assert gate <= fitted, (
        f"acceptance gate {gate} exceeds the fitted precision floor {fitted} — "
        f"the gate can never pass")


def test_decile_counts_agree_across_configs(cfg):
    """The API tiers deciles; the model builds them. A mismatch mis-tiers alerts."""
    assert cfg["model"]["priority"]["n_deciles"] == cfg["serving"]["decile_tiering"]["n_deciles"]


def test_tiering_actions_are_within_range(cfg):
    tier = cfg["serving"]["decile_tiering"]
    n = tier["n_deciles"]
    for key in tier["actions"]:
        assert 1 <= int(key) <= n, f"decile {key} outside 1..{n}"
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


def test_configured_dataset_files_are_named_consistently(cfg):
    data = cfg["model"]["data"]
    assert data["dataset"].endswith(".parquet")
    assert data["manifest"].endswith(".json")
    assert data["selected"].endswith(".json")
    assert data["screen_audit"].endswith(".csv")


def test_drift_monitor_watches_features_the_model_has(selected):
    """A drift config pointing at dropped columns monitors nothing."""
    with open(SLOT_ROOT / "mlops" / "drift_config.yaml", encoding="utf-8") as fh:
        drift = yaml.safe_load(fh)
    watched = set(drift["feature_drift"]["top_features"])
    assert watched <= set(selected["features"]), (
        f"drift_config watches features the model does not use: "
        f"{watched - set(selected['features'])}")


def test_drift_gates_match_the_acceptance_gates(cfg):
    """The monitor and the promotion gate must enforce the same FN ceiling."""
    with open(SLOT_ROOT / "mlops" / "drift_config.yaml", encoding="utf-8") as fh:
        drift = yaml.safe_load(fh)
    assert drift["performance"]["fn_rate"]["target_max"] == cfg["model"]["acceptance"]["fn_rate_max"]
    assert (drift["performance"]["precision_at_operating_point"]["floor"]
            == cfg["model"]["operating_point"]["min_precision"])


def test_config_dir_holds_exactly_the_expected_files():
    stems = sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))
    assert stems == ["feature_config.yaml", "model_config.yaml", "serving_config.yaml"]


def test_every_config_is_valid_yaml_and_documented():
    """Convention: each config opens with a comment banner explaining what drives it."""
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert yaml.safe_load(text), f"{path.name} is empty or invalid"
        assert text.lstrip().startswith("#"), f"{path.name} has no header comment"
