"""
tests/test_pipeline.py
======================
Smoke + correctness tests.  Run with:

    pytest -q

Why this matters for a portfolio: it proves the repo is engineered, not just a
notebook dumped on GitHub.  CI (GitHub Actions) runs this on every push.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.alert_system import AlertEngine
from src.evaluation import compute_metrics, tune_threshold
from src.feature_engineering import build_ai4i_features, stream_feature_columns


# ---------------------------------------------------------------- fixtures ---
@pytest.fixture(scope="module")
def small_stream():
    from src.simulate_iot_sensors import simulate_fleet
    return simulate_fleet(n_machines=4, n_cycles=250, save=False)


@pytest.fixture(scope="module")
def ai4i_df():
    from src.data_loader import load_ai4i
    from src.preprocessing import clean_ai4i
    return clean_ai4i(load_ai4i(auto_download=False))


# ------------------------------------------------------------------ tests ----
def test_config_paths_exist():
    assert config.MODELS_DIR.exists() and config.IMAGES_DIR.exists()
    assert len(config.AI4I_FEATURES) == 15


def test_simulator_physics(small_stream):
    df = small_stream
    assert len(df) == 4 * 250
    assert df["machine_id"].nunique() == 4
    # sensors must be inside physical range
    assert df["vibration_mm_s"].min() >= 0
    assert df["temperature_c"].max() < 250
    assert df["health_index"].between(0, 1).all()
    # failing samples must be measurably more extreme than healthy ones
    healthy = df.loc[df.failure_now == 0, "vibration_mm_s"].mean()
    failing = df.loc[df.failure_now == 1, "vibration_mm_s"].mean()
    if len(df[df.failure_now == 1]):
        assert failing > healthy
    # the predictive label must be >= the instantaneous label
    assert (df[config.STREAM_TARGET] >= df["failure_now"]).all()


def test_cleaning_removes_duplicates(ai4i_df):
    assert ai4i_df["udi"].is_unique
    assert ai4i_df[config.AI4I_SENSOR_COLUMNS].isna().sum().sum() == 0
    assert ai4i_df[config.AI4I_TARGET].isin([0, 1]).all()


def test_ai4i_features_are_physical(ai4i_df):
    f = build_ai4i_features(ai4i_df)
    for col in config.AI4I_ENGINEERED_COLUMNS:
        assert col in f.columns, f"missing engineered feature {col}"
        assert np.isfinite(f[col]).all(), f"{col} has inf/NaN"
    # power = torque x omega  -> must be positive
    assert (f["mechanical_power_w"] > 0).all()


def test_stream_features_have_no_leakage(small_stream):
    from src.feature_engineering import build_stream_features
    f = build_stream_features(small_stream)
    cols = stream_feature_columns(f)
    for banned in ["failure_now", config.STREAM_TARGET, "health_index", "fault_mode"]:
        assert banned not in cols
    assert len(cols) > 40


def test_metrics_and_threshold_logic():
    y = np.array([0] * 95 + [1] * 5)
    prob = np.concatenate([np.random.default_rng(0).uniform(0, 0.3, 95),
                           np.random.default_rng(1).uniform(0.6, 1.0, 5)])
    m = compute_metrics(y, prob, 0.5)
    assert m["recall"] == 1.0 and m["tp"] == 5
    assert 0 <= m["roc_auc"] <= 1
    thr, table = tune_threshold(y, prob, mode="cost")
    assert 0.05 <= thr <= 0.95 and len(table) > 10


def test_alert_engine_severity_and_dedup():
    eng = AlertEngine(verbose=False)
    assert eng.process("M1", 0.10) is None                    # healthy -> no alert
    a1 = eng.process("M1", 0.60)                              # HIGH
    assert a1 is not None and a1.severity == "HIGH"
    assert eng.process("M1", 0.62) is None                    # duplicate suppressed
    a2 = eng.process("M1", 0.95)                              # escalation allowed
    assert a2 is not None and a2.severity == "CRITICAL"
    assert a2.work_order_id.startswith("WO-")


def test_saved_model_bundle_matches_feature_contract():
    """If the model exists, it must load and predict with the right columns."""
    import joblib
    if not config.STREAM_MODEL_PATH.exists():
        pytest.skip("model not trained yet")
    bundle = joblib.load(config.STREAM_MODEL_PATH)
    assert 0.05 <= bundle["threshold"] <= 0.95
    assert bundle["metrics"]["roc_auc"] > 0.5
    assert len(bundle["feature_columns"]) > 40


def test_api_health_route():
    """Flask app must build and answer /health without crashing."""
    pytest.importorskip("flask")
    sys.path.insert(0, str(ROOT / "api"))
    from api.app import create_app
    app = create_app()
    client = app.test_client()
    r = client.get("/health")
    assert r.status_code == 200 and "status" in r.get_json()


def test_api_predict_routes():
    pytest.importorskip("flask")
    from api.app import create_app
    if not config.BEST_MODEL_PATH.exists():
        pytest.skip("model not trained yet")
    client = create_app().test_client()
    payload = {"air_temp_k": 302.0, "process_temp_k": 312.0,
               "rotational_speed_rpm": 1250, "torque_nm": 68.0, "tool_wear_min": 240}
    body = client.post("/predict", json=payload).get_json()
    assert 0 <= body["failure_probability"] <= 1
    assert body["severity"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert "verdict" in body
