"""
simulate_iot_sensors.py
=======================
VIRTUAL FACTORY SIMULATOR  ->  data/raw/iot_sensor_stream.csv

WHY THIS FILE EXISTS
--------------------
A student does not have a factory. But a predictive-maintenance model needs
*time-series* sensor data from machines that slowly degrade and eventually fail.
This module creates that data with a physically-motivated degradation model, so
the project is fully reproducible on any laptop with ZERO hardware.

WHAT IT SIMULATES (per machine)
-------------------------------
  Device      : IoT-instrumented motor / pump / compressor unit
  Sensors     : temperature(C), vibration(mm/s), current(A), pressure(bar),
                rpm, acoustic emission(dB)   <- same sensors used on a real rig
  Sampling    : every SAMPLING_MINUTES (default 10 min)  -> a real IoT edge feed

DEGRADATION PHYSICS MODELLED
----------------------------
  1. Every machine has a hidden HEALTH INDEX h(t) in [0, 1] (1 = new, 0 = dead).
  2. h decays slowly (normal wear) and then ACCELERATES after a random fault
     onset point (this is the classic "knee" of bearing/gear degradation).
  3. Sensors react DIFFERENTLY and are COUPLED, which is what makes the ML
     problem learnable and realistic:
        - temperature  rises with degradation + ambient + load
        - vibration    grows exponentially (vibration ∝ e^(k(1-h)))
        - current      rises (friction / loss of efficiency)
        - pressure     drops (internal leakage, blocked filter)
        - rpm          sags slightly under load
        - acoustic     rises sharply near end of life
  4. Scheduled maintenance resets h -> healthy (a repair), so the data contains
     many healthy->degrading->repaired cycles, exactly like a real plant.
  5. If a machine is not serviced, it FAILS, is repaired, and life restarts.

LABELS CREATED
--------------
  failure_now            : 1 on the cycle the machine actually breaks  (detection)
  failure_within_horizon : 1 if the machine will break within the next
                           FAILURE_HORIZON_CYCLES -> this is the PREDICTIVE target.
  health_index           : ground-truth hidden state (only the simulator knows it)
  fault_mode             : human-readable failure mode for the alert message
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import banner, get_logger, timer

log = get_logger("simulate")


# ---------------------------------------------------------------------------
# Helper: make one machine's life-cycle sequence
# ---------------------------------------------------------------------------
def _simulate_machine(machine_id: int, n_cycles: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate ONE IoT machine for `n_cycles` samples. Returns a DataFrame."""
    # --- 1. As-built / healthy baselines (each unit is slightly different) ----
    base_temp     = rng.uniform(62.0, 72.0)     # deg C
    base_vib      = rng.uniform(1.2, 2.4)       # mm/s RMS
    base_current  = rng.uniform(8.0, 13.0)      # A
    base_pressure = rng.uniform(4.6, 6.4)       # bar
    base_rpm      = rng.uniform(1450.0, 1495.0) # rpm
    base_acoustic = rng.uniform(68.0, 76.0)     # dB

    # --- 2. Failure mode of this unit (affects which sensor reacts most) -----
    fault_mode = rng.choice(
        ["bearing_wear", "overheating", "electrical_fault", "lubrication_loss"]
    )
    mode_gain = {
        "bearing_wear":    {"vib": 1.35, "temp": 0.85, "curr": 0.95, "pres": 0.80, "ac": 1.30, "life": 1.0},
        "overheating":     {"vib": 0.80, "temp": 1.45, "curr": 1.10, "pres": 0.90, "ac": 0.85, "life": 0.9},
        "electrical_fault":{"vib": 0.85, "temp": 1.15, "curr": 1.50, "pres": 1.00, "ac": 0.90, "life": 0.8},
        "lubrication_loss":{"vib": 1.20, "temp": 1.05, "curr": 1.05, "pres": 0.70, "ac": 1.10, "life": 1.1},
    }[fault_mode]

    # --- 3. Degradation parameters ------------------------------------------
    life_expectancy = int(rng.uniform(200, 520) * mode_gain["life"])  # cycles to failure
    fault_onset     = rng.uniform(0.45, 0.72)     # health index where damage accelerates
    wear_speed      = 1.0 / life_expectancy
    pm_prob         = 0.0015                       # chance/min of a planned service reset

    health, cycles_since_pm = 1.0, 0
    rows = []

    for c in range(n_cycles):
        # -------- degradation dynamics --------
        if health > fault_onset:
            health -= wear_speed * rng.uniform(0.85, 1.15)               # slow wear
        else:
            accel = (fault_onset - health) / max(fault_onset, 1e-6)
            health -= wear_speed * (1.0 + 9.0 * accel) * rng.uniform(0.85, 1.20)  # knee
        health = max(health, 0.02)
        dmg = 1.0 - health                      # 0 = healthy, 1 = dead

        # -------- load & environment (shift pattern + day/night thermal cycle)
        hour        = (c * config.SAMPLING_MINUTES / 60.0) % 24.0
        shift_load  = 1.0 + 0.12 * np.sin(2 * np.pi * (hour - 6) / 24.0)
        ambient     = 24.0 + 6.0 * np.sin(2 * np.pi * (hour - 15) / 24.0)
        load        = np.clip(shift_load * rng.normal(1.0, 0.02), 0.85, 1.25)

        # -------- sensor physics (non-linear in damage) --------
        knee = dmg ** 2.2                                   # accelerates near failure
        temperature = (base_temp
                       + 0.55 * (ambient - 26.0)
                       + (7.0 + 42.0 * knee) * dmg * mode_gain["temp"]
                       + rng.normal(0, 0.55))
        vibration   = (base_vib * np.exp(2.35 * dmg) * mode_gain["vib"]
                       + rng.normal(0, 0.10))
        current     = (base_current * (1 + 0.42 * dmg) * load * mode_gain["curr"]
                       + rng.normal(0, 0.16))
        pressure    = (base_pressure * (1 - 0.30 * dmg * mode_gain["pres"])
                       - rng.exponential(0.02))
        rpm         = (base_rpm * (1 - 0.035 * dmg) - 25.0 * knee
                       + rng.normal(0, 4.5))
        acoustic    = (base_acoustic + 26.0 * knee * mode_gain["ac"]
                       + 3.0 * dmg + rng.normal(0, 0.9))

        # -------- terminal failure event (sudden spike on the broken cycle) ---
        failure_now = int(dmg >= 0.93)
        if failure_now:
            vibration  *= 2.1
            temperature += 9.0
            current    *= 1.35
            acoustic   += 12.0

        # -------- occasional sensor glitch / missing value (real IoT reality) --
        if rng.random() < 0.004:
            temperature = np.nan

        rows.append({
            "machine_id": f"MOTOR-{machine_id:03d}",
            "cycle": c,
            "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=c * config.SAMPLING_MINUTES),
            "temperature_c": round(float(temperature), 2),
            "vibration_mm_s": round(float(vibration), 3),
            "current_a": round(float(current), 3),
            "pressure_bar": round(float(pressure), 3),
            "rpm": round(float(rpm), 1),
            "acoustic_db": round(float(acoustic), 2),
            "health_index": round(float(health), 4),
            "fault_mode": fault_mode,
            "failure_now": failure_now,
            "load_factor": round(float(load), 3),
            "ambient_temp_c": round(float(ambient), 2),
        })

        # -------- maintenance & repair logic --------
        if failure_now:                      # breakdown -> emergency repair
            health, cycles_since_pm = 1.0, 0
        elif rng.random() < pm_prob:         # planned preventive service
            health, cycles_since_pm = 1.0, 0
        else:
            cycles_since_pm += 1

    df = pd.DataFrame(rows)

    # Cycles elapsed since the last repair / preventive service (a real IoT KPI)
    prev_health = df.groupby("machine_id")["health_index"].shift(1).fillna(1.0)
    df["_reset"] = ((df["health_index"] >= 0.999) & (prev_health < 0.999)).astype(int)
    reset_id = df.groupby("machine_id")["_reset"].cumsum()
    df["cycles_since_maintenance"] = df.groupby(["machine_id", reset_id]).cumcount()
    df.drop(columns=["_reset"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def simulate_fleet(n_machines: int = 30,
                   n_cycles: int = 900,
                   seed: int = config.RANDOM_SEED,
                   save: bool = True) -> pd.DataFrame:
    """Simulate a whole fleet of IoT machines and (optionally) save the feed."""
    rng = np.random.default_rng(seed)
    frames = []
    for m in range(1, n_machines + 1):
        frames.append(_simulate_machine(m, n_cycles, rng))

    df = pd.concat(frames, ignore_index=True).sort_values(
        ["machine_id", "cycle"]).reset_index(drop=True)

    # ---- PREDICTIVE LABEL: will this machine fail within the next H cycles? --
    horizon = config.FAILURE_HORIZON_CYCLES
    future_fail = (
        df.groupby("machine_id")["failure_now"]
          .transform(lambda s: s[::-1].rolling(horizon, min_periods=1).max()[::-1])
    )
    df[config.STREAM_TARGET] = future_fail.astype(int)
    # 'failure_now' is already the detection label
    df["cycles_to_failure"] = (
        df.groupby("machine_id")["failure_now"]
          .transform(lambda s: s[::-1].cumsum()[::-1])
    )

    if save:
        df.to_csv(config.STREAM_RAW_CSV, index=False)
        log.info("Saved virtual IoT feed -> %s  (%d rows)", config.STREAM_RAW_CSV, len(df))
    return df


def main() -> None:
    banner("VIRTUAL IoT FACTORY SIMULATOR")
    with timer("simulation", log):
        df = simulate_fleet(n_machines=30, n_cycles=900)
    print(df.head(8).to_string(index=False))
    print("\nRows:", len(df),
          "| machines:", df['machine_id'].nunique(),
          "| step-ahead failure rate:",
          f"{df[config.STREAM_TARGET].mean()*100:.2f}%",
          "| actual breakdowns:", int(df['failure_now'].sum()))
    print("\nSensor summary:")
    print(df[config.STREAM_SENSOR_COLUMNS].describe().T[["mean", "std", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
