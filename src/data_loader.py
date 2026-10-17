"""
data_loader.py
==============
Everything that touches RAW data lives here.

Two data sources are supported - this is the "hybrid" strategy of the project:

  (A) REAL INDUSTRY DATA  : AI4I 2020 Predictive Maintenance dataset (UCI)
                            10 000 rows, 5 sensor channels, 3.4% failures,
                            5 labelled failure modes (TWF/HDF/PWF/OSF/RNF).
  (B) VIRTUAL IoT FEED    : data/raw/iot_sensor_stream.csv produced by our
                            simulator -> 27 000 rows, 30 machines, 118 real
                            breakdowns, time-series sensor streams.

If the AI4I file is missing the loader tries to download it; if there is no
internet it falls back to a synthetic AI4I-like table so the pipeline NEVER
crashes (a very handy trick for reproducible student projects).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger, print_table

log = get_logger("data")

AI4I_URL = ("https://archive.ics.uci.edu/static/public/601/"
            "ai4i+2020+predictive+maintenance+dataset.zip")


# ---------------------------------------------------------------------------
# AI4I 2020 - real dataset
# ---------------------------------------------------------------------------
def download_ai4i(force: bool = False) -> Path:
    """Download the AI4I 2020 dataset from the UCI repository if not present."""
    target = config.AI4I_RAW_CSV
    if target.exists() and not force:
        return target
    log.info("Downloading AI4I 2020 dataset from UCI ...")
    with urllib.request.urlopen(AI4I_URL, timeout=60) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(name) as f, open(target, "wb") as out:
            out.write(f.read())
    log.info("Saved -> %s", target)
    return target


def _synthetic_ai4i(n: int = 10_000, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Fallback: statistically similar stand-in if the real file cannot be used."""
    log.warning("Falling back to a SYNTHETIC AI4I-like table (no internet?).")
    rng = np.random.default_rng(seed)
    air = rng.normal(300.0, 2.0, n)
    proc = air + rng.normal(10.0, 1.1, n)
    rpm = rng.normal(1540, 180, n)
    torque = rng.normal(40, 10, n)
    wear = rng.integers(0, 250, n)
    strain = torque * (2 * np.pi * rpm / 60.0)
    prob = 1 / (1 + np.exp(-(0.00035 * strain + 0.006 * (proc - air) - 3.4)))
    fail = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({
        "UDI": np.arange(1, n + 1), "Product ID": [f"X{i:05d}" for i in range(n)],
        "Type": rng.choice(list("LMH"), n, p=[0.6, 0.3, 0.1]),
        "Air temperature [K]": air, "Process temperature [K]": proc,
        "Rotational speed [rpm]": rpm, "Torque [Nm]": torque,
        "Tool wear [min]": wear, "Machine failure": fail,
        "TWF": 0, "HDF": 0, "PWF": 0, "OSF": fail, "RNF": 0,
    })


def load_ai4i(auto_download: bool = True) -> pd.DataFrame:
    """
    Load the REAL AI4I 2020 dataset with clean pythonic column names.

    Returns
    -------
    pd.DataFrame with columns renamed via config.AI4I_COLUMN_MAP
    """
    path = config.AI4I_RAW_CSV
    if not path.exists():
        if auto_download:
            try:
                path = download_ai4i()
            except Exception as exc:                      # pragma: no cover
                log.warning("Download failed (%s)", exc)
        if not path.exists():
            return _synthetic_ai4i().rename(columns=config.AI4I_COLUMN_MAP)

    # utf-8-sig removes the BOM present in the UCI csv
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=config.AI4I_COLUMN_MAP)
    return df


# ---------------------------------------------------------------------------
# Virtual IoT sensor stream
# ---------------------------------------------------------------------------
def load_stream(auto_simulate: bool = True) -> pd.DataFrame:
    """Load the virtual IoT sensor stream (generating it first if needed)."""
    if not config.STREAM_RAW_CSV.exists():
        if not auto_simulate:
            raise FileNotFoundError(config.STREAM_RAW_CSV)
        log.info("Sensor stream not found - running the simulator ...")
        from src.simulate_iot_sensors import simulate_fleet
        simulate_fleet()
    df = pd.read_csv(config.STREAM_RAW_CSV, parse_dates=["timestamp"])
    return df


# ---------------------------------------------------------------------------
# Small EDA helper used by the CLI, notebooks and the report
# ---------------------------------------------------------------------------
def summarise(df: pd.DataFrame, name: str, target: str | None = None) -> dict:
    """Print + return a compact dataset summary (great for screenshots)."""
    info = {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    print_table(pd.DataFrame([info]).T.rename(columns={0: "value"}), f"{name} summary")
    print("\nDtypes / non-null counts:")
    print(df.dtypes.to_frame("dtype").assign(non_null=df.notna().sum()).to_string())

    if target and target in df.columns:
        vc = df[target].value_counts().sort_index()
        info["target"] = target
        info["positive_rate_pct"] = round(100 * df[target].mean(), 4)
        info["imbalance_ratio"] = f"1 : {round((1 - df[target].mean()) / max(df[target].mean(), 1e-9))}"
        print(f"\nTarget '{target}' distribution:\n{vc.to_string()}")
        print(f"Positive rate: {info['positive_rate_pct']}%  (imbalance {info['imbalance_ratio']})")
    return info
