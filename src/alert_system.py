"""
alert_system.py
===============
Where the AI stops being a Jupyter notebook and becomes an operational tool.

A prediction is useless on its own; a maintenance team needs a WORK ORDER:

    probability -> severity band -> recommended action -> alert record -> notification

Features
--------
  * Risk banding  : LOW / MEDIUM / HIGH / CRITICAL (config.RISK_BANDS)
  * De-duplication: one machine spamming the same alarm every cycle is noise.
                    Alerts are suppressed unless the severity ESCALATES or the
                    cooldown has expired.  (Real CMMS integrations do exactly this.)
  * Alert log     : every alert appended to outputs/live_alerts.csv (audit trail)
  * Notifier      : console (always) + optional webhook POST (Slack/Teams/email
                    gateway).  No credentials needed -> works offline for students.
  * Edge action   : can command a relay/motor stop for CRITICAL alerts
                    (see hardware/ for the Raspberry Pi version).

Usage
-----
    >>> from src.alert_system import AlertEngine
    >>> engine = AlertEngine()
    >>> engine.process(machine_id="MOTOR-007", probability=0.91, context={"vibration_mm_s": 21.4})
    {'machine_id': 'MOTOR-007', 'severity': 'CRITICAL', 'probability': 0.91, ...}
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger, risk_from_probability

log = get_logger("alerts")

SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


@dataclass
class Alert:
    """One maintenance alert - the unit of work handed to the plant crew."""
    timestamp: str
    machine_id: str
    severity: str
    probability: float
    recommended_action: str
    evidence: dict = field(default_factory=dict)
    acknowledged: bool = False
    work_order_id: str = ""

    def to_row(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "machine_id": self.machine_id,
            "severity": self.severity,
            "probability": round(self.probability, 4),
            "recommended_action": self.recommended_action,
            "work_order_id": self.work_order_id,
            "acknowledged": self.acknowledged,
            **{f"e_{k}": v for k, v in self.evidence.items()},
        }


class AlertEngine:
    """Stateful alert engine with escalation logic and de-duplication."""

    def __init__(self, cooldown_minutes: int = 60, webhook_url: str | None = None,
                 log_path: Path = config.LIVE_ALERTS_CSV, verbose: bool = True):
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.webhook_url = webhook_url
        self.log_path = Path(log_path)
        self.verbose = verbose
        self._last: dict[str, tuple[Alert, datetime]] = {}
        self.alerts: list[Alert] = []
        self._wo_counter = 1000

    # ------------------------------------------------------------------ core
    def process(self, machine_id: str, probability: float,
                context: dict | None = None, now: datetime | None = None) -> Alert | None:
        """
        Evaluate one prediction and return an Alert if it deserves human action.
        Returns None when suppressed (healthy, de-duplicated or cooldown).

        Suppression rules (industry practice - avoid alarm fatigue):
          1. severity LOW                 -> no alert
          2. same machine + same severity within cooldown -> no alert
          3. severity ESCALATED           -> alert anyway (critical overrides cooling)
        """
        now = now or datetime.now()
        severity, action = risk_from_probability(probability, config.RISK_BANDS)

        if severity == "LOW":
            return None

        prev = self._last.get(machine_id)
        if prev is not None:
            prev_alert, prev_time = prev
            escalated = SEVERITY_ORDER[severity] > SEVERITY_ORDER[prev_alert.severity]
            cooling = (now - prev_time) < self.cooldown
            if not escalated and cooling:
                return None                       # suppressed duplicate

        self._wo_counter += 1
        alert = Alert(
            timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
            machine_id=machine_id,
            severity=severity,
            probability=float(probability),
            recommended_action=action,
            evidence=dict(context or {}),
            work_order_id=f"WO-{self._wo_counter}",
        )
        self._last[machine_id] = (alert, now)
        self.alerts.append(alert)

        if self.verbose:
            icon = {"CRITICAL": "[!!!]", "HIGH": "[!]", "MEDIUM": "[.]"}[severity]
            print(f"{icon} ALERT {alert.work_order_id} | {severity:<8} | {machine_id} | "
                  f"P(failure)={probability:.2f} | {action}")

        self._notify(alert)
        return alert

    # ------------------------------------------------------------- reporting
    def _notify(self, alert: Alert) -> None:
        """Optional webhook (Slack/Teams/Zapier). Silent no-op if not configured."""
        if not self.webhook_url:
            return
        payload = json.dumps({
            "text": (f"*{alert.severity}* predictive-maintenance alert on "
                     f"{alert.machine_id}\nP(failure)={alert.probability:.2f}\n"
                     f"Action: {alert.recommended_action}\nWork order: {alert.work_order_id}")
        }).encode()
        try:
            req = urllib.request.Request(self.webhook_url, data=payload,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:                                   # pragma: no cover
            log.warning("Webhook failed (%s)", exc)

    def save(self) -> pd.DataFrame:
        """Persist the alert log to CSV (audit trail / dashboard input)."""
        df = pd.DataFrame([a.to_row() for a in self.alerts])
        if not df.empty:
            df.to_csv(self.log_path, index=False)
            log.info("%d alerts written -> %s", len(df), self.log_path)
        return df

    def summary(self) -> dict:
        if not self.alerts:
            return {"total_alerts": 0}
        df = pd.DataFrame([a.to_row() for a in self.alerts])
        return {
            "total_alerts": int(len(df)),
            "by_severity": df["severity"].value_counts().to_dict(),
            "machines_flagged": int(df["machine_id"].nunique()),
            "critical": int((df["severity"] == "CRITICAL").sum()),
        }


# ---------------------------------------------------------------------------
# Replay a whole sensor history through the engine (offline demo / proof)
# ---------------------------------------------------------------------------
def replay_history(predictions: pd.DataFrame | None = None,
                   machine_id: str | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Feed saved stream predictions through the AlertEngine like a live PLC feed.
    Produces outputs/live_alerts.csv and prints a maintenance work-order list.
    """
    if predictions is None:
        predictions = pd.read_csv(config.STREAM_PRED_CSV)
    if machine_id:
        predictions = predictions[predictions.machine_id == machine_id]

    engine = AlertEngine(verbose=verbose)
    start = datetime(2025, 1, 1, 0, 0, 0)
    for _, row in predictions.sort_values(["timestamp", "machine_id"]).iterrows():
        ts = datetime.fromisoformat(str(row["timestamp"]))
        engine.process(row["machine_id"], float(row["failure_probability"]),
                       context={"health_index": row.get("health_index"),
                                "fault_mode": row.get("fault_mode"),
                                "cycle": int(row["cycle"])},
                       now=ts)
    df = engine.save()
    if df.empty:
        return df

    # ------------------------------------------------------------------ #
    # Alert effectiveness: match every ALERT to the NEXT real breakdown,  #
    # per machine, and measure the advance notice (lead time).            #
    # ------------------------------------------------------------------ #
    rows = []
    preds = predictions.sort_values(["machine_id", "cycle"])
    for mid, grp in preds.groupby("machine_id"):
        fail_cycles = grp.loc[grp.y_true == 1, "cycle"].values
        if len(fail_cycles) == 0:
            continue
        events = np.split(fail_cycles, np.where(np.diff(fail_cycles) > 1)[0] + 1)
        m_alerts = (df.loc[df.machine_id == mid, "e_cycle"].astype(float).sort_values().values
                    if "e_cycle" in df.columns else np.array([]))
        for ev in events:
            fail_c = int(ev[-1])
            # alerts raised in the run-up to this breakdown
            prior = m_alerts[(m_alerts >= fail_c - config.FAILURE_HORIZON_CYCLES * 8)
                             & (m_alerts <= fail_c)]
            lead = int(fail_c - prior.min()) if len(prior) else None
            rows.append({
                "machine_id": mid,
                "breakdown_cycle": fail_c,
                "alerts_before_breakdown": int(len(prior)),
                "first_alert_cycle": int(prior.min()) if len(prior) else None,
                "lead_time_cycles": lead,
                "predicted_in_advance": bool(lead is not None and lead > 0),
                "lead_time_hours": round(lead * config.SAMPLING_MINUTES / 60, 2)
                                   if lead is not None else None,
            })
    report = pd.DataFrame(rows).sort_values(["machine_id", "breakdown_cycle"])
    report.to_csv(config.OUTPUTS_DIR / "alert_lead_time_report.csv", index=False)

    if verbose and len(report):
        caught = report["predicted_in_advance"].mean() * 100
        avg_lead = report.loc[report["predicted_in_advance"], "lead_time_cycles"].mean()
        print("\n--- ALERT EFFECTIVENESS (held-out machines) ---")
        print(report.to_string(index=False))
        print(f"\nBreakdowns predicted in advance : {caught:.0f}% "
              f"({int(report['predicted_in_advance'].sum())}/{len(report)})")
        print(f"Average early-warning lead time : {avg_lead:.1f} cycles "
              f"({avg_lead*config.SAMPLING_MINUTES/60:.1f} h of advance notice)")
    return df


if __name__ == "__main__":
    from src.utils import banner
    banner("ALERT ENGINE - REPLAY MODE")
    df = replay_history()
    print(f"\n{len(df)} alerts generated")
