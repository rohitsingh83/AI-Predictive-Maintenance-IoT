"""
model_training.py
=================
The machine-learning engine of the project.

PIPELINE INSIDE THE PIPELINE (this is the industry pattern):
    raw features -> median imputer -> StandardScaler -> classifier
wrapped in a single scikit-learn `Pipeline`.  Benefits:
    * no train/test leakage (scaling is fitted on train folds only)
    * one object to save, load, serve in Flask and version on GitHub

MODELS COMPARED
---------------
    1. LogisticRegression      - the honest baseline every engineer must beat
    2. RandomForest            - strong, interpretable, no scaling needed
    3. ExtraTrees              - faster cousin of RF, good on noisy sensors
    4. HistGradientBoosting    - boosted trees, usually the accuracy winner

IMBALANCE HANDLING (3 independent tricks, all documented)
    1. class_weight="balanced"        - cheap, no data duplication
    2. SMOTE oversampling             - optional, if imbalanced-learn is installed
    3. threshold tuning + calibration - the deployment-level fix

Every model-selection number comes from cross-validation on the TRAINING set;
only the winner is reported on the untouched TEST set.
"""

from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.evaluation import compute_metrics, tune_threshold
from src.utils import banner, get_logger, save_json, timer

warnings.filterwarnings("ignore", category=UserWarning)
log = get_logger("train")


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def get_model_zoo(fast: bool = False) -> dict:
    """Return fresh (unfitted) estimators. `fast=True` -> smaller forests."""
    rf_trees = 120 if fast else 300
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0, solver="lbfgs"),
        "random_forest": RandomForestClassifier(
            n_estimators=rf_trees, class_weight="balanced_subsample",
            min_samples_leaf=2, max_features="sqrt", n_jobs=-1,
            random_state=config.RANDOM_SEED),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=rf_trees, class_weight="balanced_subsample",
            min_samples_leaf=2, n_jobs=-1, random_state=config.RANDOM_SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=300, l2_regularization=0.1,
            class_weight="balanced", early_stopping=True, validation_fraction=0.15,
            random_state=config.RANDOM_SEED),
    }


def build_pipeline(estimator, use_scaler: bool = True) -> Pipeline:
    """imputer -> scaler -> model, as one deployable object."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Cross-validation benchmark
# ---------------------------------------------------------------------------
def benchmark_models(X_train: pd.DataFrame, y_train: pd.Series, fast: bool = False,
                     folds: int = config.CV_FOLDS, threshold_mode: str = config.THRESHOLD_MODE):
    """
    Train every model with StratifiedKFold and collect out-of-fold metrics.
    Out-of-fold (OOF) predictions: each sample is scored by a model that never
    saw it -> honest, leak-free numbers used to pick the winner.
    """
    y_train = pd.Series(y_train).astype(int).reset_index(drop=True)
    X_train = pd.DataFrame(X_train).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_SEED)

    rows, oof_store = [], {}
    for name, est in get_model_zoo(fast).items():
        t0 = time.perf_counter()
        pipe = build_pipeline(est, use_scaler=name.startswith("logistic"))
        oof = cross_val_predict(pipe, X_train, y_train, cv=skf,
                                method="predict_proba", n_jobs=1)[:, 1]
        m = compute_metrics(y_train, oof, 0.5)
        m_thr, _ = tune_threshold(y_train, oof, mode=threshold_mode)
        m_tuned = compute_metrics(y_train, oof, m_thr)
        rows.append({
            "model": name,
            "cv_pr_auc": m["pr_auc"], "cv_roc_auc": m["roc_auc"],
            "cv_recall@0.5": m["recall"], "cv_precision@0.5": m["precision"],
            "cv_f1@0.5": m["f1"],
            "cv_recall_tuned": m_tuned["recall"], "cv_precision_tuned": m_tuned["precision"],
            "cv_f1_tuned": m_tuned["f1"], "tuned_threshold": m_thr,
            "fit_seconds": round(time.perf_counter() - t0, 1),
        })
        oof_store[name] = oof
        log.info("%-24s PR-AUC %.4f | ROC-AUC %.4f | recall@tuned %.3f | %.1fs",
                 name, m["pr_auc"], m["roc_auc"], m_tuned["recall"], rows[-1]["fit_seconds"])

    return pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True), oof_store


# ---------------------------------------------------------------------------
# Final training of the winning model
# ---------------------------------------------------------------------------
def train_final_model(X_train, y_train, X_test, y_test, model_name: str,
                      calibrate: bool = True, threshold_mode: str = config.THRESHOLD_MODE):
    """
    Fit the chosen model on the full training set, calibrate its probabilities,
    tune the alarm threshold on out-of-fold predictions and score the untouched
    test set.
    """
    est = get_model_zoo()[model_name]
    pipe = build_pipeline(est, use_scaler=model_name.startswith("logistic"))

    # --- out-of-fold probabilities -> honest threshold tuning ----------------
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    oof = cross_val_predict(pipe, X_train, y_train, cv=skf,
                            method="predict_proba", n_jobs=1)[:, 1]
    threshold, threshold_table = tune_threshold(y_train, oof, mode=threshold_mode)

    # --- fit on 100% of the training data -----------------------------------
    with timer(f"fit {model_name}", log):
        pipe.fit(X_train, y_train)

    # --- probability calibration (isotonic) ---------------------------------
    # Tree ensembles emit "lumpy" probabilities; isotonic regression makes
    # P(failure)=0.80 mean ~80% of machines at that score really fail, which is
    # what the risk bands (dashboard + alert service) rely on.
    final_model = pipe
    if calibrate:
        try:
            cal = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
            cal.fit(X_train, y_train)
            final_model = cal
            log.info("Applied isotonic probability calibration")
        except Exception as exc:                        # pragma: no cover
            log.warning("Calibration skipped (%s)", exc)

    test_prob = final_model.predict_proba(X_test)[:, 1]
    bundle = {
        "model_name": model_name,
        "pipeline": final_model,
        "threshold": threshold,
        "threshold_mode": threshold_mode,
        "feature_columns": list(X_train.columns),
        "calibrated": bool(calibrate),
        "metrics": compute_metrics(y_test, test_prob, threshold),
        "metrics_at_0.5": compute_metrics(y_test, test_prob, 0.5),
        "risk_bands": config.RISK_BANDS,
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "cv_folds": config.CV_FOLDS,
        "seed": config.RANDOM_SEED,
    }
    return bundle, oof, threshold_table, test_prob


def feature_importance(bundle, top_n: int = 20) -> pd.DataFrame:
    """Feature importances (or |coef| for linear models) from a model bundle."""
    model = bundle["pipeline"]
    inner = getattr(getattr(model, "calibrated_classifiers_", [None])[0], "estimator", model)
    if hasattr(inner, "named_steps"):
        inner = inner.named_steps["model"]
    cols = bundle["feature_columns"]
    if hasattr(inner, "feature_importances_"):
        imp = inner.feature_importances_
    elif hasattr(inner, "coef_"):
        imp = np.abs(inner.coef_).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance", "importance_pct"])
    df = (pd.DataFrame({"feature": cols, "importance": imp})
            .sort_values("importance", ascending=False).head(top_n).reset_index(drop=True))
    df["importance_pct"] = (100 * df["importance"] / df["importance"].sum()).round(2)
    return df


# ---------------------------------------------------------------------------
# Orchestration for one dataset
# ---------------------------------------------------------------------------
def run_training_pipeline(X_train, y_train, X_test, y_test, dataset_name: str,
                          model_path, metrics_path, comparison_csv, fast: bool = False,
                          threshold_mode: str = config.THRESHOLD_MODE):
    """Benchmark -> pick winner -> fit -> calibrate -> evaluate -> save artefacts."""
    banner(f"TRAINING - {dataset_name}")
    log.info("Train %s | Test %s | positive rate train %.3f%%",
             X_train.shape, X_test.shape, 100 * float(np.mean(y_train)))

    table, _ = benchmark_models(X_train, y_train, fast=fast, threshold_mode=threshold_mode)
    print("\nCross-validated model comparison (out-of-fold, sorted by PR-AUC):")
    print(table.to_string(index=False))

    best_name = table.iloc[0]["model"]
    log.info("Winner by CV PR-AUC: %s", best_name)

    bundle, oof, thr_table, test_prob = train_final_model(
        X_train, y_train, X_test, y_test, best_name, calibrate=True,
        threshold_mode=threshold_mode)

    m = bundle["metrics"]
    print(f"\nFINAL TEST-SET RESULTS ({best_name}, threshold={bundle['threshold']})")
    print(f"  PR-AUC   : {m['pr_auc']:.4f}      ROC-AUC  : {m['roc_auc']:.4f}")
    print(f"  Recall   : {m['recall']:.4f}      Precision: {m['precision']:.4f}")
    print(f"  F1       : {m['f1']:.4f}      MCC      : {m['mcc']:.4f}")
    print(f"  Confusion: TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")

    joblib.dump(bundle, model_path)
    log.info("Model bundle saved -> %s", model_path)

    save_json({
        "dataset": dataset_name, "chosen_model": best_name,
        "cv_comparison": table.to_dict(orient="records"),
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "test_metrics": m,
    }, metrics_path)
    table.to_csv(comparison_csv, index=False)

    return bundle, table, oof, test_prob
