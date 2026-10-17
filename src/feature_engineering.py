"""
feature_engineering.py
======================
The step that usually decides whether an industrial ML model works.

Two families of features are built:

  A) AI4I (design/tabular data) - DOMAIN features from maintenance physics
       * temp_difference_k      : process - air temp  (poor heat dissipation)
       * mechanical_power_w     : torque x angular velocity (real load in Watts)
       * torque_per_rpm         : overstrain tendency of the drive train
       * tool_wear_rate         : wear per unit of work done
       * wear_x_torque          : interaction -> OSF signature
       * heat_dissipation_index : normalised cooling efficiency
       * rpm_deviation          : distance from the healthy operating band
       * is_high_wear / is_low_rpm_high_torque : rule-based indicator flags

  B) Virtual IoT stream (time-series) - WINDOW features per machine
       For every sensor channel and every window w in {5, 15} cycles:
          *_mean_w, *_std_w, *_trend_w (change rate), *_range_w
       Plus:
          *_delta_1        : instantaneous change (rate of change)
          *_ratio_baseline: current value / commissioning baseline
          *_zscore_baseline: how many sigma away from its own healthy baseline

WHY WINDOWS MATTER:  a single vibration sample of 7 mm/s is ambiguous; a
*rising* vibration with growing 15-cycle std is a failing bearing.  Time-series
features are what turn raw IoT telemetry into predictive signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger

log = get_logger("features")


# ===========================================================================
# A. AI4I domain features
# ===========================================================================
def build_ai4i_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Heat dissipation: difference between process and air temperature.
    #    Small difference = heat is not leaving the machine -> HDF risk.
    df["temp_difference_k"] = df["process_temp_k"] - df["air_temp_k"]

    # 2. Mechanical power (Watt): P = torque (N*m) x angular velocity (rad/s)
    df["mechanical_power_w"] = df["torque_nm"] * (2 * np.pi * df["rotational_speed_rpm"] / 60.0)

    # 3. Torque per rpm - high value = the drive train strains at low speed
    df["torque_per_rpm"] = df["torque_nm"] / df["rotational_speed_rpm"].replace(0, np.nan)

    # 4. Tool wear per unit of work already performed
    df["tool_wear_rate"] = df["tool_wear_min"] / (df["mechanical_power_w"] / 1000.0 + 1e-6)

    # 5. Interaction term: a worn tool + heavy torque is the classic OSF recipe
    df["wear_x_torque"] = df["tool_wear_min"] * df["torque_nm"]

    # 6. Cooling efficiency ratio
    df["heat_dissipation_index"] = df["temp_difference_k"] / (df["mechanical_power_w"] / 1000.0 + 1e-6)

    # 7. rpm distance from the healthy band (AI4I healthy rpm ~ 1300-1700)
    df["rpm_deviation"] = (df["rotational_speed_rpm"] - 1500.0).abs()

    # 8-9. Domain rule flags maintenance engineers actually use
    df["is_high_wear"] = (df["tool_wear_min"] > 200).astype(int)
    df["is_low_rpm_high_torque"] = ((df["rotational_speed_rpm"] < 1380) &
                                    (df["torque_nm"] > 55)).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    df[config.AI4I_ENGINEERED_COLUMNS] = df[config.AI4I_ENGINEERED_COLUMNS].fillna(0.0)
    log.info("build_ai4i_features: %d engineered features added", len(config.AI4I_ENGINEERED_COLUMNS))
    return df


# ===========================================================================
# B. Virtual stream window features
# ===========================================================================
def _baseline_reference(df: pd.DataFrame, cycles: int = 40) -> pd.DataFrame:
    """
    Per-machine 'as-new' baseline = median of the first `cycles` samples.
    A real edge gateway learns this during commissioning; here we emulate it.
    """
    sensors = config.STREAM_SENSOR_COLUMNS
    early = df[df["cycle"] < cycles]
    base = early.groupby("machine_id")[sensors].median()
    return base.add_suffix("_baseline")


def build_stream_features(df: pd.DataFrame,
                          windows=config.STREAM_WINDOWS,
                          add_baseline: bool = True) -> pd.DataFrame:
    """
    Turn the raw IoT stream into a supervised, time-series feature table.

    All rolling statistics use only PAST + CURRENT samples (no centred windows),
    so this feature table could be produced live on an edge device.
    """
    df = df.copy()
    sensors = config.STREAM_SENSOR_COLUMNS
    g = df.groupby("machine_id", sort=False)

    new_cols: dict[str, pd.Series] = {}

    # ---- 1. rolling statistics (mean / std / range / trend) per window ------
    for w in windows:
        for c in sensors:
            rolling = g[c].rolling(w, min_periods=1)
            new_cols[f"{c}_mean_{w}"] = rolling.mean().reset_index(level=0, drop=True)
            new_cols[f"{c}_std_{w}"] = rolling.std().reset_index(level=0, drop=True)
            new_cols[f"{c}_max_{w}"] = rolling.max().reset_index(level=0, drop=True)
            new_cols[f"{c}_min_{w}"] = rolling.min().reset_index(level=0, drop=True)
        # trend = net change across the window  (fast, leakage-free slope proxy)
        for c in sensors:
            new_cols[f"{c}_trend_{w}"] = df[c] - g[c].shift(w - 1)

    # ---- 2. instantaneous rate of change ------------------------------------
    for c in sensors:
        new_cols[f"{c}_delta_1"] = g[c].diff()

    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    # range features (max-min over window)
    for w in windows:
        for c in sensors:
            df[f"{c}_range_{w}"] = df[f"{c}_max_{w}"] - df[f"{c}_min_{w}"]

    # ---- 3. baseline-relative features (own-healthy-reference) --------------
    if add_baseline:
        base = _baseline_reference(df)
        df = df.merge(base, left_on="machine_id", right_index=True, how="left")
        for c in sensors:
            b = df[f"{c}_baseline"].replace(0, np.nan)
            df[f"{c}_ratio_baseline"] = (df[c] / b).replace([np.inf, -np.inf], np.nan)
            df[f"{c}_zscore_baseline"] = ((df[c] - df[f"{c}_baseline"]) /
                                          df.groupby("machine_id")[c].transform("std")
                                          .replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # ---- 4. composite health proxies (physics-inspired) ---------------------
    df["vibration_x_temp"] = df["vibration_mm_s"] * df["temperature_c"]
    df["power_proxy_w"] = df["current_a"] * 230.0            # single-phase ~230 V
    df["efficiency_index"] = df["pressure_bar"] / (df["current_a"] + 1e-6)
    df["acoustic_x_vibration"] = df["acoustic_db"] * df["vibration_mm_s"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.loc[:, ~df.columns.duplicated()]
    log.info("build_stream_features: table is now %d columns", df.shape[1])
    return df


def stream_feature_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric column that is a legal MODEL INPUT (no leakage allowed)."""
    leakage = {
        "cycle", "health_index", "health_pct", "fault_mode",
        "failure_now", config.STREAM_TARGET, "cycles_to_failure",
    }
    exclude = leakage | {"machine_id", "timestamp"}
    cols = [c for c in df.columns
            if c not in exclude
            and df[c].dtype.kind in "if"
            and not c.endswith("_baseline")]
    # '_baseline' columns are per-machine constants -> also leakage-ish for a
    # machine-wise split, so they are excluded from the model inputs.
    return sorted(cols)


def select_top_features(X: pd.DataFrame, y: pd.Series, k: int = 30):
    """Return (top_k_columns, full_mi_score_series)."""
    """Quick variance-sane top-k selection using Mutual Information."""
    from sklearn.feature_selection import mutual_info_classif
    Xc = X.fillna(0.0)
    mi = mutual_info_classif(Xc, y, random_state=config.RANDOM_SEED)
    order = np.argsort(mi)[::-1][:k]
    scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    log.info("Top-%d features by MI: %s", k, list(scores.index[:10]))
    return list(Xc.columns[order]), scores
