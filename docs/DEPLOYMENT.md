# Deployment Guide

Three deployment levels, from "laptop demo" to "edge + cloud". Each is real and
runnable; pick the one your demo needs.

## Level 1 - Laptop / local (what a reviewer will do)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py all            # simulate -> train -> evaluate -> figures -> dashboard
python api/app.py             # REST API on http://0.0.0.0:5000
```

Interactive pieces:
* `outputs/dashboard.html` - open in a browser (self-contained, works offline)
* `python realtime_simulator.py` - fake a live factory into the API
* `python main.py demo` - 30-second console demo, ideal for screen recording

## Level 2 - Docker (one command, any machine)

```bash
docker compose up --build
# API  -> http://localhost:5000/health
# dashboard -> outputs/dashboard.html is mounted to the host
```

## Level 3 - Edge + cloud (industry architecture)

```
   Raspberry Pi node                Cloud / on-prem server
┌──────────────────────┐        ┌───────────────────────────────┐
│ hardware/raspberry_  │  MQTT  │  broker (Mosquitto/IoT Core)  │
│ pi_sensor_node.py    │ ─────► │  → Flask API (container)      │
│ (buffers + acts)     │        │  → model retraining job       │
└──────────────────────┘        │  → alerts → CMMS/Slack/e-mail │
                                └───────────────────────────────┘
```

Practical notes learned from real deployments:

1. **Do not retrain on the edge.** Push telemetry up, retrain on a server, ship a new
   model file down (version it - the bundle records `trained_at`).
2. **Cache a baseline.** The per-machine baseline z-scores in `feature_engineering.py`
   are computed from commissioning data; store them per machine, not globally.
3. **Alarm fatigue kills adoption.** The de-duplication + escalation logic in
   `alert_system.py` matters more than a 1% accuracy gain.
4. **Always keep the human in the loop** for CRITICAL alerts.
5. **Log everything** - `outputs/live_alerts.csv`, `outputs/test_predictions.csv` and
   the API's `/metrics` endpoint are your audit trail when someone asks
   "why did you stop my machine?".
