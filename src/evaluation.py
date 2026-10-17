"""
evaluation.py
=============
Turning predictions into numbers a maintenance manager actually cares about.

Why NOT just accuracy?
----------------------
Failures are rare (3% in AI4I, 2.6% in our virtual fleet).  A lazy model that
always says "healthy" scores 97% accuracy and is 100% useless.  So we report:

  * Precision        - of the alarms we raise, how many were real?
  * Recall           - of the real failures, how many did we catch?  <-- safety
  * F1               - balance of the two
  * ROC-AUC          - ranking quality over all thresholds
  * PR-AUC (AP)      - the honest metric for imbalanced data (our main metric)
  * MCC / balanced accuracy - robustness under imbalance
  * Confusion matrix - TP / FP / FN / TN in plain English
  * Brier score      - are the probabilities trustworthy? (matters for risk bands)
  * Cost model       - what saved/downtime dollars the model is worth
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve)

from src import config
from src.utils import get_logger

log = get_logger("eval")


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Full metric pack for one model + one decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out = {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_samples": int(len(y_true)),
        "n_positives": int(y_true.sum()),
        "missed_failures": int(fn),
        "false_alarms": int(fp),
    }
    return out


def tune_threshold(y_true, y_prob, mode: str = "cost") -> tuple[float, dict]:
    """
    Choose the alarm threshold DELIBERATELY instead of blindly using 0.5.

    mode='f1'   -> maximise F1
    mode='cost' -> minimise expected business cost:
                     missed failure  : unplanned downtime (very expensive)
                     false alarm     : one extra inspection (cheap)
                   This is exactly how real PdM business cases are built.
    """
    y_true = np.asarray(y_true).astype(int)
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, t)
        missed_cost = m["fn"] * config.MISSED_FAILURE_COST
        alarm_cost = m["fp"] * config.FALSE_ALARM_COST
        rows.append({"threshold": t, "f1": m["f1"], "recall": m["recall"],
                     "precision": m["precision"], "expected_cost_usd": missed_cost + alarm_cost,
                     "total_cost_usd": missed_cost + alarm_cost + m["tp"] * config.PLANNED_MAINTENANCE_COST})
    table = pd.DataFrame(rows)
    if mode == "f1":
        best = table.loc[table["f1"].idxmax()]
    else:
        best = table.loc[table["expected_cost_usd"].idxmin()]
    log.info("Tuned threshold (mode=%s) -> %.2f | recall %.2f precision %.2f",
             mode, best["threshold"], best["recall"], best["precision"])
    return float(best["threshold"]), table


def business_impact(m: dict, n_machines_in_fleet: int = 30) -> dict:
    """
    Translate confusion-matrix numbers into money.

    Scenario: one unplanned breakdown = AVG_UNPLANNED_DOWNTIME_HOURS of lost
    production at DOWNTIME_COST_PER_HOUR; a planned repair costs
    PLANNED_MAINTENANCE_COST; an extra AI-triggered inspection costs
    INSPECTION_COST.  Reactive (run-to-failure) maintenance is the baseline.
    """
    downtime_cost = config.MISSED_FAILURE_COST

    # Baseline: no AI -> every failure becomes unplanned downtime
    baseline_cost = (m["tp"] + m["fn"]) * downtime_cost
    # With AI: caught failures -> planned repair; missed -> downtime; alarms -> inspections
    ai_cost = (m["tp"] * config.PLANNED_MAINTENANCE_COST
               + m["fn"] * downtime_cost
               + m["fp"] * config.FALSE_ALARM_COST)
    saved = baseline_cost - ai_cost

    return {
        "cost_of_one_unplanned_breakdown_usd": downtime_cost,
        "baseline_run_to_failure_cost_usd": round(baseline_cost, 2),
        "cost_with_ai_usd": round(ai_cost, 2),
        "net_saving_usd": round(saved, 2),
        "saving_pct": round(100 * saved / max(baseline_cost, 1e-9), 2),
        "failures_prevented": int(m["tp"]),
        "failures_missed": int(m["fn"]),
        "false_alarms": int(m["fp"]),
        "cost_per_false_alarm_usd": round(config.FALSE_ALARM_COST, 2),
        "cost_per_missed_failure_usd": round(config.MISSED_FAILURE_COST, 2),
        "lives_of_equipment_protected": int(m["tp"]) + int(m["fn"]),
        "sensor_kit_cost_usd": round(n_machines_in_fleet * config.SENSOR_KIT_COST, 2),
        "payback_ratio": round(saved / max(n_machines_in_fleet * config.SENSOR_KIT_COST, 1e-9), 2),
    }


def classification_text_report(y_true, y_pred, labels=("Healthy", "Failure")) -> str:
    return classification_report(y_true, y_pred, target_names=list(labels), digits=3)


def curve_data(y_true, y_prob) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}
