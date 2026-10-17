"""
live_inference.py
=================
The bridge between "a trained model file" and "a live IoT feed".

A production PdM service never receives a tidy feature vector - it receives a
stream of raw sensor readings.  This module:

  1. keeps a rolling buffer of recent readings per machine (like an edge gateway)
  2. rebuilds EXACTLY the same features the model was trained on
  3. calls the model and returns a rich, human-readable verdict

Used by  api/app.py  (Flask)  and  src/dashboard.py  (HTML dashboard).

    >>> from src.live_inference import PredictiveMaintenanceService
    >>> svc = PredictiveMaintenanceService()
    >>> svc.predict_stream(history_df)      # pandas DataFrame of recent readings
    {'machine_id': ..., 'failure_probability': 0.91, 'severity': 'CRITICAL', ...}
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.feature_engineering import build_stream_features, stream_feature_columns
from src.utils import get_logger, risk_from_probability

log = get_logger("inference")


@dataclass
class Prediction:
    machine_id: str
    failure_probability: float
    will_fail: bool
    threshold: float
    severity: str
    recommended_action: str
    top_signals: list
    model_name: str
    model_version: str

    def to_dict(self) -> dict:
        return asdict(self)


class PredictiveMaintenanceService:
    """Loads a saved bundle ONCE and serves predictions cheaply."""

    def __init__(self, model_path: Path = config.STREAM_MODEL_PATH,
                 ai4i_path: Path = config.BEST_MODEL_PATH):
        self.model_path = Path(model_path)
        self.ai4i_path = Path(ai4i_path)
        self.bundle = None
        self.ai4i_bundle = None
        self.buffers: dict[str, deque] = {}
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        if self.model_path.exists():
            self.bundle = joblib.load(self.model_path)
            log.info("Stream model loaded: %s (threshold %.2f, %d features)",
                     self.bundle["model_name"], self.bundle["threshold"],
                     len(self.bundle["feature_columns"]))
        else:
            log.warning("Stream model not found at %s - run train_model.py", self.model_path)
        if self.ai4i_path.exists():
            self.ai4i_bundle = joblib.load(self.ai4i_path)
            log.info("AI4I model loaded: %s (threshold %.2f)",
                     self.ai4i_bundle["model_name"], self.ai4i_bundle["threshold"])

    @property
    def ready(self) -> bool:
        return self.bundle is not None

    def info(self) -> dict:
        if not self.ready:
            return {"ready": False,
                    "hint": "Run `python train_model.py` to create models/*.pkl"}
        b = self.bundle
        return {
            "ready": True,
            "model_name": b["model_name"],
            "task": f"P(failure within {config.FAILURE_HORIZON_CYCLES} cycles / "
                    f"{config.FAILURE_HORIZON_CYCLES*config.SAMPLING_MINUTES/60:.1f} h)",
            "alarm_threshold": b["threshold"],
            "threshold_mode": b.get("threshold_mode", "cost"),
            "n_features": len(b["feature_columns"]),
            "calibrated_probabilities": b["calibrated"],
            "trained_at": b["trained_at"],
            "test_metrics": b["metrics"],
            "risk_bands": [{"from": t, "severity": s, "action": a}
                           for t, s, a in config.RISK_BANDS],
            "ai4i_model": None if self.ai4i_bundle is None else {
                "model_name": self.ai4i_bundle["model_name"],
                "threshold": self.ai4i_bundle["threshold"],
                "metrics": self.ai4i_bundle["metrics"],
            },
        }

    # --------------------------------------------------- feature reconstruction
    def _align(self, feat_row: pd.DataFrame) -> pd.DataFrame:
        """Guarantee the exact column set/order the model expects."""
        cols = self.bundle["feature_columns"]
        return feat_row.reindex(columns=cols).astype(float)

    def _top_signals(self, feat_row: pd.DataFrame, k: int = 5) -> list[dict]:
        """Explain a prediction with the strongest baseline-relative deviations."""
        signals = []
        for sensor in config.STREAM_SENSOR_COLUMNS:
            z_col, r_col = f"{sensor}_zscore_baseline", f"{sensor}_ratio_baseline"
            z = feat_row.get(z_col, pd.Series([np.nan])).iloc[0]
            r = feat_row.get(r_col, pd.Series([np.nan])).iloc[0]
            if pd.notna(z) and abs(z) > 1.5:
                signals.append({
                    "signal": sensor,
                    "deviation_sigma": round(float(z), 2),
                    "vs_baseline_ratio": round(float(r), 3) if pd.notna(r) else None,
                })
        signals.sort(key=lambda s: abs(s["deviation_sigma"]), reverse=True)
        return signals[:k]

    # ------------------------------------------------------------ public APIs
    def predict_stream(self, history: pd.DataFrame, machine_id: str | None = None) -> dict:
        """
        Predict from a window of RAW sensor readings (oldest -> newest).

        `history` must contain the STREAM_SENSOR_COLUMNS.  A single row works
        (features degrade gracefully), but >= 16 readings give full accuracy.
        """
        if not self.ready:
            raise RuntimeError("Stream model not loaded - run `python train_model.py` first")

        df = history.copy()
        if "machine_id" not in df.columns:
            df["machine_id"] = machine_id or "LIVE-001"
        if "cycle" not in df.columns:
            df["cycle"] = np.arange(len(df))
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.Timestamp.now()
        for c in config.STREAM_SENSOR_COLUMNS:
            if c not in df.columns:
                raise ValueError(f"missing required sensor column: {c}")

        min_history = max(config.STREAM_WINDOWS)
        enough = len(df) >= min_history
        feat = build_stream_features(df, add_baseline=enough)
        feat_row = feat.tail(1)
        X = self._align(feat_row)

        prob = float(self.bundle["pipeline"].predict_proba(X)[0, 1])
        thr = float(self.bundle["threshold"])
        severity, action = risk_from_probability(prob, config.RISK_BANDS)

        return Prediction(
            machine_id=str(feat_row["machine_id"].iloc[0]),
            failure_probability=round(prob, 4),
            will_fail=bool(prob >= thr),
            threshold=round(thr, 4),
            severity=severity,
            recommended_action=action,
            top_signals=self._top_signals(feat_row),
            model_name=self.bundle["model_name"],
            model_version=self.bundle["trained_at"],
        ).to_dict()

    def predict_ai4i(self, readings: dict) -> dict:
        """
        Score one AI4I-style machine snapshot (the Flask /predict endpoint).
        Accepts either the raw AI4I names or the short dashboard names.
        """
        if self.ai4i_bundle is None:
            raise RuntimeError("AI4I model not loaded - run `python train_model.py`")

        aliases = {
            "air_temp_k": ["air_temp_k", "air_temperature_k", "air_temp", "Air temperature [K]"],
            "process_temp_k": ["process_temp_k", "process_temperature_k", "process_temp",
                               "Process temperature [K]"],
            "rotational_speed_rpm": ["rotational_speed_rpm", "rpm", "Rotational speed [rpm]"],
            "torque_nm": ["torque_nm", "torque", "Torque [Nm]"],
            "tool_wear_min": ["tool_wear_min", "tool_wear", "Tool wear [min]"],
        }
        row = {}
        for canonical, names in aliases.items():
            for n in names:
                if n in readings:
                    row[canonical] = float(readings[n])
                    break
            else:
                raise ValueError(f"missing field: {canonical} (accepted: {names})")
        row["type"] = str(readings.get("type", readings.get("Type", "M"))).upper()
        row["type_encoded"] = config.AI4I_TYPE_MAP.get(row["type"], 1)

        from src.feature_engineering import build_ai4i_features
        feat = build_ai4i_features(pd.DataFrame([row]))
        X = feat.reindex(columns=self.ai4i_bundle["feature_columns"]).astype(float)

        prob = float(self.ai4i_bundle["pipeline"].predict_proba(X)[0, 1])
        thr = float(self.ai4i_bundle["threshold"])
        severity, action = risk_from_probability(prob, config.RISK_BANDS)
        explain = {
            "mechanical_power_w": round(float(feat["mechanical_power_w"].iloc[0]), 1),
            "temp_difference_k": round(float(feat["temp_difference_k"].iloc[0]), 2),
            "tool_wear_min": round(float(feat["tool_wear_min"].iloc[0]), 1),
            "torque_nm": round(float(feat["torque_nm"].iloc[0]), 1),
            "rotational_speed_rpm": round(float(feat["rotational_speed_rpm"].iloc[0]), 1),
        }
        return {
            "failure_probability": round(prob, 4),
            "will_fail": bool(prob >= thr),
            "threshold": round(thr, 4),
            "severity": severity,
            "recommended_action": action,
            "engineered_features": explain,
            "model_name": self.ai4i_bundle["model_name"],
            "model_version": self.ai4i_bundle["trained_at"],
            "verdict": ("MACHINE FAILURE PREDICTED - schedule maintenance now"
                        if prob >= thr else "Machine is running normally"),
        }

    # ------------------------------------------------------- streaming buffer
    def push(self, machine_id: str, reading: dict, window: int = 24) -> dict:
        """Edge-gateway style API: push one reading, get a verdict back."""
        buf = self.buffers.setdefault(machine_id, deque(maxlen=window))
        buf.append(reading)
        hist = pd.DataFrame(list(buf))
        hist["machine_id"] = machine_id
        return self.predict_stream(hist, machine_id=machine_id)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.utils import banner
    banner("LIVE INFERENCE SELF-TEST")
    svc = PredictiveMaintenanceService()
    stream = pd.read_csv(config.STREAM_RAW_CSV)
    # take the healthy start and the failing end of one machine
    for label, part in [("healthy window", stream[(stream.machine_id == "MOTOR-001") & (stream.cycle < 20)]),
                        ("late-life window", stream[(stream.machine_id == "MOTOR-001")].sort_values(
                            "failure_now", ascending=False).head(20))]:
        out = svc.predict_stream(part.sort_values("cycle"))
        print(f"\n[{label}] P(failure)={out['failure_probability']:.3f} "
              f"-> {out['severity']} | {out['recommended_action']}")
        if out["top_signals"]:
            print("   top signals:", out["top_signals"])
