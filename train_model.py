"""
train_model.py
==============
ONE command trains everything:

    python train_model.py               # full training (both datasets)
    python train_model.py --fast        # small forests, quick smoke test
    python train_model.py --dataset ai4i
    python train_model.py --dataset stream

It trains TWO models (this is what makes the project "industry-oriented"):

  MODEL 1 - AI4I 2020 real factory data  -> tabular failure classification
            (5 sensor channels + 9 engineered physics features)
  MODEL 2 - Virtual IoT sensor stream    -> predictive, time-series early warning
            (92 window/rolling/baseline features, machine-wise split)

Artefacts written:
    models/predictive_maintenance_model.pkl   (AI4I, served by the Flask API)
    models/ai4i_metrics.json
    models/stream_failure_model.pkl           (time-series early-warning model)
    models/stream_metrics.json
    reports/model_comparison*.csv
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

# allow "python train_model.py" from the repo root
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from src import config
from src.data_loader import load_ai4i, load_stream
from src.feature_engineering import build_ai4i_features, build_stream_features, stream_feature_columns
from src.model_training import run_training_pipeline
from src.preprocessing import clean_ai4i, clean_stream, split_ai4i, split_stream_by_machine
from src.utils import banner, get_logger, json_safe

log = get_logger("main")


# ---------------------------------------------------------------------------
def train_ai4i(fast: bool, threshold_mode: str = config.THRESHOLD_MODE):
    """Real-industry tabular pipeline (AI4I 2020)."""
    df = clean_ai4i(load_ai4i())
    df = build_ai4i_features(df)
    train, test = split_ai4i(df)
    X_tr, y_tr = train[config.AI4I_FEATURES], train[config.AI4I_TARGET]
    X_te, y_te = test[config.AI4I_FEATURES], test[config.AI4I_TARGET]
    bundle, table, oof, prob = run_training_pipeline(
        X_tr, y_tr, X_te, y_te, "AI4I-2020 (real factory data)",
        config.BEST_MODEL_PATH, config.AI4I_METRICS, config.MODEL_COMPARISON_CSV, fast, threshold_mode)
    # persist test predictions for plotting / reporting
    preds = test[["udi", "product_id", "type"]].copy()
    preds["y_true"] = y_te.values
    preds["failure_probability"] = prob
    preds["prediction"] = (prob >= bundle["threshold"]).astype(int)
    from src.utils import risk_from_probability
    preds["severity"] = [risk_from_probability(p, config.RISK_BANDS)[0] for p in prob]
    preds["recommended_action"] = [risk_from_probability(p, config.RISK_BANDS)[1] for p in prob]
    preds.to_csv(config.PREDICTIONS_CSV, index=False)
    log.info("Test predictions saved -> %s", config.PREDICTIONS_CSV)
    return bundle


def train_stream(fast: bool, threshold_mode: str = config.THRESHOLD_MODE):
    """Time-series early-warning pipeline (virtual IoT fleet)."""
    df = clean_stream(load_stream())
    df = build_stream_features(df)
    train, test, held_out = split_stream_by_machine(df)
    cols = stream_feature_columns(df)
    X_tr, y_tr = train[cols], train[config.STREAM_TARGET]
    X_te, y_te = test[cols], test[config.STREAM_TARGET]
    bundle, table, oof, prob = run_training_pipeline(
        X_tr, y_tr, X_te, y_te, "Virtual IoT stream (30 machines)",
        config.STREAM_MODEL_PATH, config.STREAM_METRICS,
        config.REPORTS_DIR / "model_comparison_stream.csv", fast, threshold_mode)

    preds = test[["machine_id", "cycle", "timestamp", "health_index", "fault_mode"]].copy()
    preds["y_true"] = y_te.values
    preds["failure_probability"] = prob
    preds["prediction"] = (prob >= bundle["threshold"]).astype(int)
    preds.to_csv(config.STREAM_PRED_CSV, index=False)

    # remember which machines the model never saw -> used by the dashboard
    from src.utils import save_json
    save_json({"held_out_machines": held_out, "feature_count": len(cols)},
              config.REPORTS_DIR / "split_info.json")
    log.info("Stream predictions saved -> %s", config.STREAM_PRED_CSV)
    return bundle


def main():
    ap = argparse.ArgumentParser(description="Train the predictive-maintenance models")
    ap.add_argument("--dataset", choices=["ai4i", "stream", "both"], default="both")
    ap.add_argument("--fast", action="store_true", help="smaller forests / quicker CV")
    ap.add_argument("--threshold-mode", choices=["cost", "f1"], default=config.THRESHOLD_MODE,
                    help="how to pick the alarm threshold: business cost or F1")
    args = ap.parse_args()

    banner("AI-POWERED PREDICTIVE MAINTENANCE - MODEL TRAINING")
    print(f"  dataset : {args.dataset}\n  fast    : {args.fast}\n  seed    : {config.RANDOM_SEED}")

    results = {}
    if args.dataset in ("ai4i", "both"):
        b1 = train_ai4i(args.fast, args.threshold_mode)
        results["ai4i"] = b1["metrics"]
    if args.dataset in ("stream", "both"):
        b2 = train_stream(args.fast, args.threshold_mode)
        results["stream"] = b2["metrics"]

    print("\n" + "=" * 78)
    print("TRAINING COMPLETE")
    print("=" * 78)
    print(json.dumps(json_safe(results), indent=2)[:2000])
    print("\nNext step:  python main.py dashboard")


if __name__ == "__main__":
    main()
