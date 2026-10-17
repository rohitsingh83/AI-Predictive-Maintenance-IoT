"""
visualization.py
================
All the figures that make the GitHub repo look like an industrial dashboard.

Figures produced (written to images/):
  01_dataset_overview.png        - class balance + sensor distributions vs failure
  02_sensor_correlation.png      - correlation heatmap (AI4I) - feature insight
  03_model_comparison.png        - cross-validated benchmark of the 4 models
  04_confusion_matrix.png        - test-set confusion matrix with business meaning
  05_roc_pr_curves.png           - ROC + Precision-Recall curves (imbalanced data)
  06_feature_importance.png      - what the model actually learned
  07_threshold_optimisation.png  - precision / recall / cost vs alarm threshold
  08_probability_distribution.png- separation of healthy vs failing machines
  09_degradation_trajectory.png  - virtual machine wearing out (sensor time-series)
  10_early_warning_timeline.png  - AI alarm 6 cycles BEFORE the breakdown
  11_fleet_health_map.png        - current health of all 30 virtual machines
  12_eda_stream_overview.png     - virtual IoT stream EDA
"""

from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")                      # headless -> works on servers/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.evaluation import compute_metrics, curve_data, tune_threshold
from src.utils import get_logger

log = get_logger("viz")

# ---------------------------------------------------------------------------
# Global house style (consistent colours across every figure)
# ---------------------------------------------------------------------------
PALETTE = {"healthy": "#2E86AB", "failure": "#E4572E", "accent": "#F18F01",
           "ok": "#3FA34D", "warn": "#F18F01", "crit": "#C81D25", "grid": "#B9C4C9"}
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": config.PLOT_DPI, "savefig.dpi": config.PLOT_DPI,
    "savefig.bbox": "tight", "font.size": 11, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.edgecolor": "#5A6B73",
    "figure.facecolor": "white", "axes.labelcolor": "#24343B",
    "text.color": "#24343B",
})


def _save(fig, name: str) -> str:
    path = config.IMAGES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    log.info("figure -> %s", path.name)
    return str(path)


def _load_bundle(path):
    return joblib.load(path) if path.exists() else None


# ===========================================================================
# 1. EDA - AI4I
# ===========================================================================
def plot_dataset_overview(ai4i: pd.DataFrame) -> str:
    sensors = config.AI4I_SENSOR_COLUMNS
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    # (a) class balance
    counts = ai4i[config.AI4I_TARGET].value_counts().sort_index()
    bars = axes[0].bar(["Healthy (0)", "Failure (1)"], counts.values,
                       color=[PALETTE["healthy"], PALETTE["failure"]], width=0.6)
    for b, v in zip(bars, counts.values):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({100*v/len(ai4i):.2f}%)",
                     ha="center", va="bottom", fontweight="bold")
    axes[0].set_title("(a) Class balance - imbalanced by design")
    axes[0].set_ylabel("Samples"); axes[0].set_ylim(0, counts.max() * 1.25)

    # (b-e) sensor distributions split by class
    for ax, col in zip(axes[1:5], sensors):
        for lab, colour, name in [(0, PALETTE["healthy"], "Healthy"),
                                  (1, PALETTE["failure"], "Failure")]:
            sub = ai4i.loc[ai4i[config.AI4I_TARGET] == lab, col]
            sns.kdeplot(sub, ax=ax, fill=True, alpha=0.45, color=colour,
                        label=name, warn_singular=False)
        ax.set_title(f"({chr(98+list(sensors).index(col))}) {col}")
        ax.set_xlabel(""); ax.legend(fontsize=9)

    # (f) failure modes
    mode_counts = ai4i[config.AI4I_MODE_COLUMNS].sum().sort_values(ascending=False)
    sns.barplot(x=mode_counts.values, y=[m.replace("_", " ").upper() for m in mode_counts.index],
                ax=axes[5], color=PALETTE["accent"])
    axes[5].set_title("(f) Labelled failure modes"); axes[5].set_xlabel("Occurrences")

    fig.suptitle("AI4I 2020 Predictive Maintenance - Exploratory Data Analysis",
                 fontsize=17, fontweight="bold", y=1.02)
    return _save(fig, "01_dataset_overview.png")


# ===========================================================================
# 2. Correlation heatmap
# ===========================================================================
def plot_correlation(ai4i_feat: pd.DataFrame) -> str:
    cols = config.AI4I_SENSOR_COLUMNS + config.AI4I_ENGINEERED_COLUMNS[:6] + [config.AI4I_TARGET]
    corr = ai4i_feat[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
                linewidths=0.6, cbar_kws={"shrink": 0.75}, annot_kws={"size": 8}, ax=ax)
    ax.set_title("Correlation matrix - raw sensors vs engineered physics features\n"
                 "(look at the bottom row: engineered features correlate with failure)", pad=16)
    return _save(fig, "02_sensor_correlation.png")


# ===========================================================================
# 3. Model comparison
# ===========================================================================
def plot_model_comparison(table: pd.DataFrame, dataset: str, fname: str) -> str:
    t = table.sort_values("cv_pr_auc", ascending=True)
    metrics = [("cv_pr_auc", "PR-AUC (main)"), ("cv_roc_auc", "ROC-AUC"),
               ("cv_f1_tuned", "F1 @ tuned thr"), ("cv_recall_tuned", "Recall @ tuned thr")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6.5), sharey=True)
    colours = [PALETTE["crit"] if n == t.iloc[-1]["model"] else PALETTE["healthy"]
               for n in t["model"]]
    for ax, (col, title) in zip(axes, metrics):
        bars = ax.barh(t["model"].str.replace("_", " "), t[col], color=colours)
        for b, v in zip(bars, t[col]):
            ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                    va="center", fontsize=10, fontweight="bold")
        ax.set_title(title); ax.set_xlim(0, 1.12); ax.set_xlabel("score")
    axes[0].set_ylabel("")
    fig.suptitle(f"Model benchmark - {dataset}\n(5-fold stratified cross-validation, "
                 "red bar = selected model)", fontsize=15, fontweight="bold")
    return _save(fig, fname)


# ===========================================================================
# 4. Confusion matrix
# ===========================================================================
def plot_confusion(m: dict, title: str, fname: str) -> str:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=False, square=True,
                linewidths=2, linecolor="white", ax=ax)
    labels = [["True Negative\n(machine healthy\n- correctly ignored)",
               "False Alarm\n(extra inspection)"],
              ["MISSED FAILURE\n(unplanned downtime!)",
               "True Positive\n(failure predicted in time)"]]
    for i in range(2):
        for j in range(2):
            colour = "white" if cm[i, j] > cm.max() * 0.55 else "#24343B"
            ax.text(j + 0.5, i + 0.42, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=22, fontweight="bold", color=colour)
            ax.text(j + 0.5, i + 0.72, labels[i][j], ha="center", va="center",
                    fontsize=9.5, color=colour)
    ax.set_xticklabels(["Predicted\nHealthy", "Predicted\nFailure"])
    ax.set_yticklabels(["Actually\nHealthy", "Actually\nFailure"], rotation=0)
    ax.set_title(f"{title}\nAccuracy {m['accuracy']*100:.2f}% | Precision {m['precision']:.3f} | "
                 f"Recall {m['recall']:.3f} | F1 {m['f1']:.3f} | ROC-AUC {m['roc_auc']:.3f}\n"
                 f"threshold = {m['threshold']}", pad=14)
    return _save(fig, fname)


# ===========================================================================
# 5. ROC + PR curves
# ===========================================================================
def plot_roc_pr(y_true, y_prob, title: str, fname: str) -> str:
    c = curve_data(y_true, y_prob)
    m = compute_metrics(y_true, y_prob, 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    axes[0].plot(c["fpr"], c["tpr"], color=PALETTE["healthy"], lw=2.5,
                 label=f"Model (AUC = {m['roc_auc']:.4f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1.2, label="Random guess (0.5)")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC curve"); axes[0].legend(loc="lower right")

    baseline = float(np.mean(y_true))
    axes[1].plot(c["recall"], c["precision"], color=PALETTE["failure"], lw=2.5,
                 label=f"Model (PR-AUC = {m['pr_auc']:.4f})")
    axes[1].axhline(baseline, ls="--", color="grey", lw=1.2,
                    label=f"No-skill baseline = {baseline:.3f}")
    axes[1].set_xlabel("Recall (failures caught)"); axes[1].set_ylabel("Precision (alarms that were real)")
    axes[1].set_title("Precision-Recall curve (honest metric for rare failures)")
    axes[1].legend(loc="lower left")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    return _save(fig, fname)


# ===========================================================================
# 6. Feature importance
# ===========================================================================
def plot_feature_importance(imp: pd.DataFrame, dataset: str, fname: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 8))
    top = imp.head(18).iloc[::-1]
    colours = [PALETTE["accent"] if any(k in f for k in
               ("vibration", "temp", "wear", "torque", "acoustic", "pressure"))
               else PALETTE["healthy"] for f in top["feature"]]
    bars = ax.barh(top["feature"], top["importance"], color=colours)
    for b, v in zip(bars, top["importance"]):
        ax.text(v + top["importance"].max() * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_title(f"Feature importance - {dataset}\n"
                 "(orange = physical sensor channels, blue = engineered/composite features)")
    ax.set_xlabel("importance")
    return _save(fig, fname)


# ===========================================================================
# 7. Threshold optimisation
# ===========================================================================
def plot_thresholds(thr_table: pd.DataFrame, threshold: float, dataset: str, fname: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    eff = thr_table[(thr_table["threshold"] >= 0.05) & (thr_table["threshold"] <= 0.95)]

    for col, colour, label in [("recall", PALETTE["crit"], "Recall (failures caught)"),
                               ("precision", PALETTE["healthy"], "Precision (alarm quality)"),
                               ("f1", PALETTE["ok"], "F1")]:
        axes[0].plot(eff["threshold"], eff[col], lw=2.2, color=colour, label=label)
    axes[0].axvline(threshold, ls="--", color="#24343B", lw=1.6,
                    label=f"chosen threshold = {threshold:.2f}")
    axes[0].set_xlabel("Alarm threshold"); axes[0].set_ylabel("score"); axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Precision / Recall trade-off"); axes[0].legend(fontsize=9)

    axes[1].plot(eff["threshold"], eff["expected_cost_usd"] / 1000, lw=2.4, color=PALETTE["accent"],
                 label="Expected cost of errors (k$)")
    axes[1].plot(eff["threshold"], eff["total_cost_usd"] / 1000, lw=2.0, ls=":",
                 color=PALETTE["healthy"], label="Total maintenance cost (k$)")
    axes[1].axvline(threshold, ls="--", color="#24343B", lw=1.6,
                    label=f"cost-optimal = {threshold:.2f}")
    axes[1].set_xlabel("Alarm threshold"); axes[1].set_ylabel("cost (thousand USD)")
    axes[1].set_title("Business cost model\n(miss = 8 h downtime; false alarm = 0.5 h stop + triage)")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"Why 0.5 is a bad default - threshold optimisation on {dataset}",
                 fontsize=15, fontweight="bold")
    return _save(fig, fname)


# ===========================================================================
# 8. Probability separation
# ===========================================================================
def plot_probability_distribution(y_true, y_prob, threshold: float, dataset: str, fname: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.histplot(y_prob[np.asarray(y_true) == 0], bins=40, ax=ax, color=PALETTE["healthy"],
                 alpha=0.6, label="Healthy machines", stat="count", log_scale=(False, True))
    sns.histplot(y_prob[np.asarray(y_true) == 1], bins=40, ax=ax, color=PALETTE["failure"],
                 alpha=0.7, label="Failing machines", stat="count", log_scale=(False, True))
    ax.axvline(threshold, ls="--", color="#24343B", lw=2,
               label=f"alarm threshold = {threshold:.2f}")
    ax.set_xlabel("Predicted failure probability"); ax.set_ylabel("count (log scale)")
    ax.set_title(f"Probability separation on the untouched test set - {dataset}")
    ax.legend()
    return _save(fig, fname)


# ===========================================================================
# 9-11. Virtual fleet figures
# ===========================================================================
def plot_degradation_trajectory(stream: pd.DataFrame, fname: str = "09_degradation_trajectory.png") -> str:
    """Pick a machine that really fails and plot its sensors wearing out."""
    fails = stream.groupby("machine_id")["failure_now"].sum()
    if fails.max() == 0:
        mid, lo, hi = stream["machine_id"].iloc[0], 0, 400
    else:
        mid = fails.idxmax()
        fail_cycle = int(stream.loc[(stream.machine_id == mid) & (stream.failure_now == 1), "cycle"].iloc[0])
        lo, hi = max(0, fail_cycle - 260), fail_cycle + 15
    d = stream[(stream.machine_id == mid) & stream.cycle.between(lo, hi)]

    sensors = [("temperature_c", "Temperature (degC)", PALETTE["failure"]),
               ("vibration_mm_s", "Vibration (mm/s RMS)", PALETTE["accent"]),
               ("current_a", "Current (A)", PALETTE["healthy"]),
               ("pressure_bar", "Pressure (bar)", "#7B5EA7")]
    fig, axes = plt.subplots(2, 2, figsize=(17, 9), sharex=True)
    for ax, (col, label, colour) in zip(axes.ravel(), sensors):
        ax.plot(d["cycle"], d[col], color=colour, lw=1.9)
        ax.set_ylabel(label)
        ax.fill_between(d["cycle"], d[col], d[col].min(), alpha=0.12, color=colour)
    for ax in axes.ravel():
        ax.axvline(int(lo) + 260, color=PALETTE["crit"], ls="--", lw=1.8)
        ax.axhline(0, alpha=0)
    axes[0, 0].set_title(f"Virtual machine {mid} - degradation trajectory "
                         f"(fault mode: {d['fault_mode'].iloc[-1]})")
    axes[1, 0].set_xlabel("Cycle (10 min each)"); axes[1, 1].set_xlabel("Cycle (10 min each)")
    fig.text(0.5, -0.02, "red dashed line = actual breakdown.  Sensors drift together BEFORE the "
             "failure - that drift is what the AI learns.", ha="center", fontsize=11, style="italic")
    return _save(fig, fname)


def plot_early_warning(preds: pd.DataFrame, bundle: dict,
                       fname: str = "10_early_warning_timeline.png") -> str:
    """
    Zoom into ONE real breakdown of a machine the model never saw during
    training and show the AI raising the alarm cycles BEFORE it happens.

    Also prints the average early-warning lead time over every breakdown in the
    held-out fleet - the single most important number a maintenance manager
    cares about ("how much notice do I get?").
    """
    thr = bundle["threshold"]
    preds = preds.sort_values(["machine_id", "cycle"])
    events, lead_times = [], []

    for mid, d in preds.groupby("machine_id"):
        fail_cycles = d.loc[d.y_true == 1, "cycle"].values
        if len(fail_cycles) == 0:
            continue
        # group consecutive cycles (label horizon) into single events
        events_idx = np.split(fail_cycles, np.where(np.diff(fail_cycles) > 1)[0] + 1)
        for ev in events_idx:
            fail_c = int(ev[-1])                     # the true breakdown cycle
            window = d[d.cycle.between(fail_c - 40, fail_c)]
            alarms = window.loc[window.failure_probability >= thr, "cycle"]
            if len(alarms):
                lead = int(fail_c - alarms.min())
                lead_times.append(lead)
                events.append({"machine_id": mid, "fail_cycle": fail_c, "lead": lead,
                               "prob_before": float(d.loc[d.cycle == fail_c - 1,
                                                          "failure_probability"].iloc[0])})
            else:
                events.append({"machine_id": mid, "fail_cycle": fail_c, "lead": 0,
                               "prob_before": float(d.loc[d.cycle == fail_c - 1,
                                                          "failure_probability"].iloc[0])})

    if not events:
        log.warning("no breakdown events in predictions - skipping early-warning plot")
        return ""

    # show the breakdown that was caught with the longest (but realistic) notice
    catches = [e for e in events if e["lead"] > 0]
    chosen = (sorted(catches, key=lambda e: e["lead"], reverse=True)[1]
              if len(catches) > 2 else (catches[0] if catches else events[0]))
    mid, fail_c = chosen["machine_id"], chosen["fail_cycle"]

    d = preds[(preds.machine_id == mid) &
              preds.cycle.between(fail_c - 60, fail_c + 8)].copy()
    first_alarm = d.loc[d.failure_probability >= thr, "cycle"].min()
    stream = pd.read_csv(config.STREAM_RAW_CSV, parse_dates=["timestamp"])
    s_raw = stream[(stream.machine_id == mid) &
                   stream.cycle.between(fail_c - 60, fail_c + 8)]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1.0, 0.9]})
    ax1, ax2, ax3 = axes

    # ---- panel 1: AI probability ------------------------------------------
    ax1.plot(d["cycle"], d["failure_probability"], color=PALETTE["crit"], lw=2.6,
             label="AI failure probability (model output)")
    ax1.axhline(thr, ls="--", color="#24343B", lw=1.6,
                label=f"alarm threshold = {thr:.2f} (cost-optimised)")
    ax1.fill_between(d["cycle"], thr, d["failure_probability"],
                     where=d["failure_probability"] >= thr, color=PALETTE["crit"],
                     alpha=0.22, label="alarm active")
    ax1.set_ylabel("P(failure within\n6 cycles / 1 hour)")
    ax1.set_ylim(-0.03, 1.12); ax1.set_title(
        f"PREDICTIVE EARLY WARNING - machine {mid} (never seen during training), "
        f"breakdown at cycle {fail_c}")
    ax1.legend(loc="lower left", fontsize=9.5, framealpha=0.95)

    # ---- panel 2: the sensor signals the model actually uses ---------------
    ax2.plot(s_raw["cycle"], s_raw["vibration_mm_s"], color=PALETTE["accent"], lw=2.2,
             label="vibration (mm/s)")
    ax2.set_ylabel("Vibration (mm/s)", color=PALETTE["accent"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["accent"])
    ax2b = ax2.twinx()
    ax2b.plot(s_raw["cycle"], s_raw["temperature_c"], color=PALETTE["failure"], lw=2.2,
              label="temperature (degC)")
    ax2b.plot(s_raw["cycle"], s_raw["acoustic_db"], color="#7B5EA7", lw=1.6, ls="--",
              label="acoustic emission (dB)")
    ax2b.set_ylabel("Temperature / acoustic", color=PALETTE["failure"])
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10, framealpha=0.95)
    ax2.set_title("Raw IoT sensor channels - the physical drift the AI detects")

    # ---- panel 3: ground-truth health -------------------------------------
    ax3.plot(s_raw["cycle"], s_raw["health_index"], color=PALETTE["healthy"], lw=2.6,
             label="Ground-truth health index (simulator only - NOT a model input)")
    ax3.fill_between(s_raw["cycle"], s_raw["health_index"], 0, color=PALETTE["healthy"],
                     alpha=0.12)
    ax3.set_ylabel("Health index"); ax3.set_ylim(0, 1.08)
    ax3.set_xlabel("Cycle (each cycle = 10 minutes of machine operation)")
    ax3.legend(loc="lower left", fontsize=10, framealpha=0.95)

    for ax in axes:
        ax.axvline(fail_c, color=PALETTE["crit"], lw=2.4, ls=":")
        if not np.isnan(first_alarm) and first_alarm > fail_c - 60:
            ax.axvline(first_alarm, color=PALETTE["ok"], lw=2.2, ls="--")
    axes[0].annotate("ACTUAL BREAKDOWN", xy=(fail_c, 1.04), xytext=(fail_c - 26, 1.04),
                     color=PALETTE["crit"], fontweight="bold", fontsize=11,
                     arrowprops=dict(arrowstyle="->", color=PALETTE["crit"], lw=1.6))
    if not np.isnan(first_alarm) and first_alarm > fail_c - 60:
        axes[0].annotate("AI ALARM FIRED", xy=(first_alarm, 0.62), xytext=(first_alarm - 52, 0.62),
                         color=PALETTE["ok"], fontweight="bold", fontsize=11,
                         arrowprops=dict(arrowstyle="->", color=PALETTE["ok"], lw=1.6))

    lead = int(fail_c - first_alarm) if not np.isnan(first_alarm) else 0
    avg_lead = float(np.mean(lead_times)) if lead_times else 0.0
    caught = 100 * len(catches) / len(events)
    fig.text(0.5, -0.015,
             f"This breakdown: {lead} cycles of notice ({lead*config.SAMPLING_MINUTES/60:.1f} h)  |  "
             f"Held-out fleet: {caught:.0f}% of {len(events)} breakdowns caught, "
             f"average notice {avg_lead:.1f} cycles "
             f"({avg_lead*config.SAMPLING_MINUTES/60:.1f} h)",
             ha="center", fontsize=12.5, fontweight="bold", color=PALETTE["ok"])
    return _save(fig, fname)


def plot_fleet_health(stream: pd.DataFrame, fname: str = "11_fleet_health_map.png") -> str:
    last = stream.sort_values("cycle").groupby("machine_id").tail(1).copy()
    last["health_pct"] = last["health_index"] * 100
    last = last.sort_values("health_pct")
    colours = [PALETTE["crit"] if h < 60 else PALETTE["warn"] if h < 85 else PALETTE["ok"]
               for h in last["health_pct"]]
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(last["machine_id"], last["health_pct"], color=colours)
    ax.axhline(60, color=PALETTE["crit"], ls="--", lw=1.4)
    ax.axhline(85, color=PALETTE["warn"], ls="--", lw=1.4)
    ax.text(len(last) - 0.4, 61, " critical < 60%", color=PALETTE["crit"], fontsize=10)
    ax.text(len(last) - 0.4, 86, " monitor < 85%", color=PALETTE["warn"], fontsize=10)
    for b, v, m in zip(bars, last["health_pct"], last["machine_id"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", fontsize=8.5)
    ax.set_ylabel("Health index (%)"); ax.set_ylim(0, 108)
    ax.set_title("Fleet health snapshot - all 30 virtual IoT machines "
                 "(last simulated cycle)")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=9)
    return _save(fig, fname)


def plot_stream_eda(stream: pd.DataFrame, fname: str = "12_eda_stream_overview.png") -> str:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    for ax, col in zip(axes[:5], config.STREAM_SENSOR_COLUMNS):
        for lab, colour, name in [(0, PALETTE["healthy"], "Healthy"),
                                  (1, PALETTE["failure"], "Failing (within 6 cycles)")]:
            sns.kdeplot(stream.loc[stream[config.STREAM_TARGET] == lab, col], ax=ax,
                        fill=True, alpha=0.45, color=colour, label=name, warn_singular=False)
        ax.set_title(col); ax.set_xlabel(""); ax.legend(fontsize=8)

    # fleet-level failure statistics
    per_machine = stream.groupby("machine_id")["failure_now"].sum().sort_values(ascending=False)
    axes[5].bar(per_machine.index, per_machine.values, color=PALETTE["failure"])
    axes[5].set_title("Breakdowns per machine (simulated 150 h run)")
    axes[5].set_ylabel("breakdowns")
    plt.setp(axes[5].get_xticklabels(), rotation=90, fontsize=7)
    fig.suptitle("Virtual IoT sensor stream - EDA (27,000 samples / 30 machines / 118 breakdowns)",
                 fontsize=16, fontweight="bold", y=1.02)
    return _save(fig, fname)


# ===========================================================================
# Orchestrator
# ===========================================================================
def generate_all_figures(skip_expensive: bool = False) -> list[str]:
    """Build every figure from saved artefacts. Missing artefacts are skipped."""
    from src.data_loader import load_stream
    from src.feature_engineering import build_ai4i_features
    from src.model_training import feature_importance

    made = []
    # ---- AI4I figures ------------------------------------------------------
    try:
        from src.preprocessing import clean_ai4i
        ai4i = clean_ai4i(load_ai4i_safe())
        ai4i_f = build_ai4i_features(ai4i)
        made.append(plot_dataset_overview(ai4i))
        made.append(plot_correlation(ai4i_f))
    except Exception as exc:
        log.warning("AI4I EDA figures skipped (%s)", exc)

    # ---- model-level figures ----------------------------------------------
    if config.AI4I_METRICS.exists():
        meta = json.loads(config.AI4I_METRICS.read_text())
        table = pd.DataFrame(meta["cv_comparison"])
        made.append(plot_model_comparison(table, "AI4I 2020 (real factory data)",
                                          "03_model_comparison.png"))
        made.append(plot_confusion(meta["test_metrics"], "AI4I 2020 - confusion matrix",
                                   "04_confusion_matrix.png"))
        bundle = _load_bundle(config.BEST_MODEL_PATH)
        if bundle is not None and config.PREDICTIONS_CSV.exists():
            p = pd.read_csv(config.PREDICTIONS_CSV)
            made.append(plot_roc_pr(p["y_true"], p["failure_probability"],
                                    "AI4I 2020 - ROC & Precision-Recall", "05_roc_pr_curves.png"))
            made.append(plot_probability_distribution(
                p["y_true"], p["failure_probability"], bundle["threshold"], "AI4I 2020",
                "08_probability_distribution.png"))
            imp = feature_importance(bundle, 25)
            if not imp.empty:
                made.append(plot_feature_importance(imp, "AI4I 2020", "06_feature_importance.png"))
            # threshold table (recomputed on OOF stored in metrics if absent)
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            if not skip_expensive:
                from src.preprocessing import split_ai4i
                tr, _ = split_ai4i(ai4i_f)
                X, y = tr[config.AI4I_FEATURES], tr[config.AI4I_TARGET]
                pipe = bundle["pipeline"]
                oof = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True,
                                        random_state=config.RANDOM_SEED),
                                        method="predict_proba", n_jobs=1)[:, 1]
                thr_tab = tune_threshold(y, oof, mode=bundle.get("threshold_mode", "cost"))[1]
                made.append(plot_thresholds(thr_tab, bundle["threshold"], "AI4I 2020",
                                            "07_threshold_optimisation.png"))
    else:
        log.warning("AI4I metrics not found - run `python train_model.py` first")

    # ---- stream / virtual fleet figures ------------------------------------
    if config.STREAM_RAW_CSV.exists():
        stream = load_stream()
        made.append(plot_stream_eda(stream))
        made.append(plot_degradation_trajectory(stream))
        made.append(plot_fleet_health(stream))
    if config.STREAM_METRICS.exists() and config.STREAM_PRED_CSV.exists():
        meta_s = json.loads(config.STREAM_METRICS.read_text())
        b_s = _load_bundle(config.STREAM_MODEL_PATH)
        ps = pd.read_csv(config.STREAM_PRED_CSV)
        made.append(plot_confusion(meta_s["test_metrics"],
                                   "Virtual IoT stream - confusion matrix (unseen machines)",
                                   "13_confusion_matrix_stream.png"))
        made.append(plot_model_comparison(pd.DataFrame(meta_s["cv_comparison"]),
                                          "Virtual IoT stream (30 machines)",
                                          "14_model_comparison_stream.png"))
        if b_s:
            made.append(plot_early_warning(ps, b_s))
            made.append(plot_roc_pr(ps["y_true"], ps["failure_probability"],
                                    "Virtual IoT stream - ROC & PR (unseen machines)",
                                    "15_roc_pr_stream.png"))
            imp = feature_importance(b_s, 25)
            if not imp.empty:
                made.append(plot_feature_importance(imp, "Virtual IoT stream",
                                                    "16_feature_importance_stream.png"))
    return [m for m in made if m]


def load_ai4i_safe():
    from src.data_loader import load_ai4i
    return load_ai4i()


if __name__ == "__main__":
    from src.utils import banner
    banner("GENERATING ALL FIGURES")
    files = generate_all_figures()
    print(f"\n{len(files)} figures written to {config.IMAGES_DIR}")
