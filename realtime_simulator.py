"""
realtime_simulator.py
=====================
THE "IoT HARDWARE REPLACEMENT": a script that pretends to be a live factory.

Use it when you want to demo the system WITHOUT a Raspberry Pi:
  * it pushes one sensor reading every N seconds to the Flask API
  * after a while the machine starts degrading and the API raises work orders
  * stop it any time - the machine's wear is remembered between runs

    Terminal 1 :  python api/app.py
    Terminal 2 :  python realtime_simulator.py --machines 3 --interval 1.0

    # or without the API (prints predictions locally):
    python realtime_simulator.py --local
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1. A lightweight machine twin (the same physics as the batch simulator,
#    but running one cycle at a time so it feels "live")
# ---------------------------------------------------------------------------
class MachineTwin:
    def __init__(self, machine_id: str, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.machine_id = machine_id
        self.base_temp = rng.uniform(62, 72)
        self.base_vib = rng.uniform(1.2, 2.4)
        self.base_current = rng.uniform(8, 13)
        self.base_pressure = rng.uniform(4.6, 6.4)
        self.base_rpm = rng.uniform(1450, 1495)
        self.base_acoustic = rng.uniform(68, 76)
        self.health = 1.0
        self.wear_speed = 1.0 / rng.uniform(180, 420)
        self.fault_onset = rng.uniform(0.45, 0.72)
        self.cycle = 0

    def step(self) -> dict:
        """Advance the twin by one sampling interval and return a reading."""
        self.cycle += 1
        self.health = max(0.02, self.health - self.wear_speed * self.rng.uniform(0.85, 1.2))
        dmg = 1 - self.health
        knee = dmg ** 2.2
        reading = {
            "temperature_c": round(float(self.base_temp + 45 * knee
                                         + self.rng.normal(0, 0.55)), 2),
            "vibration_mm_s": round(float(self.base_vib * np.exp(2.35 * dmg)
                                          + self.rng.normal(0, 0.1)), 3),
            "current_a": round(float(self.base_current * (1 + 0.42 * dmg)
                                     + self.rng.normal(0, 0.16)), 3),
            "pressure_bar": round(float(self.base_pressure * (1 - 0.3 * dmg)
                                        - abs(self.rng.normal(0, 0.02))), 3),
            "rpm": round(float(self.base_rpm * (1 - 0.035 * dmg) - 25 * knee
                               + self.rng.normal(0, 4.5)), 1),
            "acoustic_db": round(float(self.base_acoustic + 26 * knee + 3 * dmg
                                       + self.rng.normal(0, 0.9)), 2),
        }
        if dmg >= 0.93:                       # breakdown -> repair
            self.health = 1.0
            self.wear_speed = 1.0 / self.rng.uniform(180, 420)
        return reading


# ---------------------------------------------------------------------------
def post(url: str, payload: dict, timeout: float = 4.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser(description="Virtual IoT edge feed")
    ap.add_argument("--machines", type=int, default=3,
                    help="how many virtual machines to run in parallel")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between sensor pushes")
    ap.add_argument("--url", default="http://127.0.0.1:5000/predict/reading")
    ap.add_argument("--local", action="store_true",
                    help="skip the API and score locally with the saved model")
    ap.add_argument("--cycles", type=int, default=0, help="stop after N cycles (0 = forever)")
    ap.add_argument("--accentuate", type=float, default=6.0,
                    help="multiply the wear rate so a demo fails fast (1 = realistic)")
    args = ap.parse_args()

    twins = [MachineTwin(f"MOTOR-{i:03d}", seed=100 + i) for i in range(1, args.machines + 1)]
    for tw in twins:                          # make the demo visible quickly
        tw.wear_speed *= args.accentuate

    svc = None
    if args.local:
        from src.live_inference import PredictiveMaintenanceService
        svc = PredictiveMaintenanceService()

    print(f"Streaming {args.machines} virtual machines every {args.interval}s "
          f"-> {'local model' if args.local else args.url}")
    print("Ctrl+C to stop.\n")

    cycle = 0
    try:
        while True:
            cycle += 1
            for tw in twins:
                reading = tw.step()
                payload = {"machine_id": tw.machine_id, **reading}
                try:
                    if args.local:
                        out = svc.push(tw.machine_id, reading)
                    else:
                        out = post(args.url, payload)
                    p = out.get("failure_probability", 0.0)
                    sev = out.get("severity", "?")
                    flag = {"CRITICAL": "!!!", "HIGH": " ! ", "MEDIUM": " . "}.get(sev, "   ")
                    wo = out.get("work_order", {}).get("id", "")
                    print(f"{flag} {datetime.now():%H:%M:%S} {tw.machine_id} "
                          f"temp={reading['temperature_c']:6.1f} vib={reading['vibration_mm_s']:6.2f} "
                          f"| P(fail)={p:0.3f} {sev:<8} {wo}")
                except Exception as exc:
                    print(f"[feed] {tw.machine_id}: {exc}")
                    time.sleep(1)
            if args.cycles and cycle >= args.cycles:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
