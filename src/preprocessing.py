"""
preprocessing.py
================
DATA CLEANING + SPLITTING.  No model, no plots - just making the data trustworthy.

Industry checklist implemented here
-----------------------------------
  1. Drop exact duplicates.
  2. Fix impossible / physically invalid values (negative vibration, rpm = 0 ...).
  3. Handle missing values (median per machine for sensors, drop if all missing).
  4. Cap extreme outliers with the IQR rule BUT never touching failure rows
     (a broken machine is *supposed* to look extreme - deleting it would be
     data leakage in reverse and destroy the whole point of the project).
  5. Split train/test.  For the time-series stream we split BY MACHINE so the
     model is evaluated on machines it has never seen -> no leakage.  This is a
     detail that separates a student repo from an industry-grade repo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger

log = get_logger("preprocess")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _iqr_cap(series: pd.Series, factor: float = 3.0) -> pd.Series:
    """Cap values outside [Q1 - f*IQR, Q3 + f*IQR] - a robust outlier treatment."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr):
        return series
    return series.clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)


def _physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce sensor ranges that a real instrument cannot violate."""
    bounds = {
        "vibration_mm_s": (0.0, None), "current_a": (0.0, None), "rpm": (0.0, None),
        "temperature_c": (-20.0, 250.0), "pressure_bar": (0.0, None),
        "acoustic_db": (0.0, 180.0),
        "air_temp_k": (250.0, 350.0), "process_temp_k": (250.0, 400.0),
        "rotational_speed_rpm": (0.0, None), "torque_nm": (0.0, None),
        "tool_wear_min": (0.0, None),
    }
    n_fixed = 0
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            before = df[col].copy()
            df[col] = df[col].clip(lower=lo, upper=hi)
            n_fixed += int((df[col] != before).sum())
    if n_fixed:
        log.info("Physical-bounds enforcement corrected %d values", n_fixed)
    return df


# ---------------------------------------------------------------------------
# AI4I cleaning
# ---------------------------------------------------------------------------
def clean_ai4i(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the AI4I 2020 table and add type_encoded."""
    df = df.copy()
    n0 = len(df)

    df = df.drop_duplicates(subset=["udi"]).drop_duplicates()
    numeric = [c for c in df.columns if df[c].dtype.kind in "if"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # sensors: median imputation (few cells, safe); labels: 0 fill
    for c in config.AI4I_SENSOR_COLUMNS:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in config.AI4I_MODE_COLUMNS + [config.AI4I_TARGET]:
        df[c] = df[c].fillna(0).astype(int)
    df["type"] = df["type"].astype(str).str.upper().str.strip()
    df["type_encoded"] = df["type"].map(config.AI4I_TYPE_MAP).fillna(1).astype(int)

    # outlier capping on SENSOR channels only, healthy rows only (see docstring)
    healthy = df[config.AI4I_TARGET] == 0
    for c in config.AI4I_SENSOR_COLUMNS:
        capped = _iqr_cap(df.loc[healthy, c])
        df.loc[healthy, c] = capped
    df = _physical_bounds(df)

    log.info("clean_ai4i: %d -> %d rows, %d cols", n0, len(df), df.shape[1])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Virtual stream cleaning
# ---------------------------------------------------------------------------
def clean_stream(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the (already tidy) simulator output for time-series modelling."""
    df = df.copy()
    n0 = len(df)
    df = df.sort_values(["machine_id", "cycle"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["machine_id", "cycle"])

    # 1. sensor drop-outs: interpolate inside each machine's own timeline
    for c in config.STREAM_SENSOR_COLUMNS:
        df[c] = (df.groupby("machine_id")[c]
                   .transform(lambda s: s.interpolate(limit_direction="both")))

    # 2. a row with no telemetry at all is useless for the model
    missing_all = df[config.STREAM_SENSOR_COLUMNS].isna().all(axis=1)
    if missing_all.any():
        log.warning("Dropping %d rows with zero telemetry", int(missing_all.sum()))
        df = df[~missing_all]

    df = _physical_bounds(df)

    # 3. simple derived telemetry used by the dashboard (not model features)
    df["temp_rise_above_ambient"] = df["temperature_c"] - df["ambient_temp_c"]
    df["health_pct"] = (df["health_index"] * 100).round(2)

    log.info("clean_stream: %d -> %d rows", n0, len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train / test splitting
# ---------------------------------------------------------------------------
def split_ai4i(df: pd.DataFrame):
    """Stratified 80/20 split for the AI4I table."""
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED,
        stratify=df[config.AI4I_TARGET],
    )
    train = train.reset_index(drop=True); test = test.reset_index(drop=True)
    log.info("AI4I split -> train %s (%d fails) | test %s (%d fails)",
             train.shape, train[config.AI4I_TARGET].sum(),
             test.shape, test[config.AI4I_TARGET].sum())
    return train, test


def split_stream_by_machine(df: pd.DataFrame, test_size: float = config.TEST_SIZE):
    """
    Machine-wise split: every machine goes entirely to train OR to test.

    WHY: if we split rows randomly, samples from the same machine appear in both
    sets and a model can simply memorise that machine's personality.  Splitting
    by machine answers the real business question - "will the AI work on a
    machine it has never seen before?"  Machines are assigned in a stratified
    order so the test set still contains breakdowns.
    """
    stats = (df.groupby("machine_id")[config.STREAM_TARGET]
               .agg(rows="size", positives="sum")
               .sort_values(["positives", "rows"], ascending=False))
    n_test_target = int(config.TEST_SIZE * len(df))
    test_machines, taken = [], 0
    for i, (mid, row) in enumerate(stats.iterrows()):
        if taken + row["rows"] <= n_test_target * 1.15 or i == 0:
            test_machines.append(mid); taken += row["rows"]
    if len(test_machines) >= len(stats):          # safety: never empty the train set
        test_machines = list(stats.index[:max(1, len(stats) // 5)])

    is_test = df["machine_id"].isin(test_machines)
    train, test = df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)

    if test[config.STREAM_TARGET].sum() < 10:     # fall back to a stratified row split
        log.warning("Machine split produced too few positives - using stratified rows")
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=test_size,
                                       random_state=config.RANDOM_SEED,
                                       stratify=df[config.STREAM_TARGET])
    log.info("Stream split -> train %d machines / %d rows | test %d machines / %d rows"
             " | test positives %d",
             train.machine_id.nunique(), len(train), test.machine_id.nunique(),
             len(test), test[config.STREAM_TARGET].sum())
    return train.reset_index(drop=True), test.reset_index(drop=True), test_machines


# ---------------------------------------------------------------------------
# One-call pipeline used by main.py / notebooks
# ---------------------------------------------------------------------------
def prepare_datasets(save: bool = True):
    """Load -> clean -> split -> (optionally) save parquet/csv. Returns a dict."""
    from src.data_loader import load_ai4i, load_stream

    ai4i_raw = load_ai4i()
    ai4i = clean_ai4i(ai4i_raw)
    tr_a, te_a = split_ai4i(ai4i)

    stream_raw = load_stream()
    stream = clean_stream(stream_raw)
    tr_s, te_s, held_out_machines = split_stream_by_machine(stream)

    if save:
        tr_a.to_csv(config.AI4I_TRAIN_CSV, index=False)
        te_a.to_csv(config.AI4I_TEST_CSV, index=False)
        tr_s.to_csv(config.STREAM_TRAIN_CSV, index=False)
        te_s.to_csv(config.STREAM_TEST_CSV, index=False)

    return {
        "ai4i_raw": ai4i_raw, "ai4i": ai4i, "ai4i_train": tr_a, "ai4i_test": te_a,
        "stream_raw": stream_raw, "stream": stream,
        "stream_train": tr_s, "stream_test": te_s,
        "held_out_machines": held_out_machines,
    }
