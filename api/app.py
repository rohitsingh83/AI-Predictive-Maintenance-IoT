"""
api/app.py
==========
Flask REST API - the "AI inference service" of the architecture.

This is the piece that lets an ESP32 / Raspberry Pi / SCADA system on the shop
floor ask the AI for a decision.

Run it:
    python api/app.py                # dev server on http://0.0.0.0:5000
    python api/app.py --port 8000

Endpoints
---------
GET  /health                 liveness + model readiness
GET  /model/info             model metadata, thresholds, metrics, risk bands
POST /predict                AI4I snapshot   -> failure probability + action
POST /predict/window         raw sensor window (iot stream) -> prediction
POST /predict/reading        single reading (edge style, stateful buffer)
GET  /alerts                 recent alerts raised by this service
GET  /metrics                service counters (for Prometheus-style monitoring)

Example
-------
    curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \\
         -d '{"air_temp_k":298.1,"process_temp_k":308.6,
              "rotational_speed_rpm":1551,"torque_nm":42.8,"tool_wear_min":0}'
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# make "import src.*" work no matter where the script is launched from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, request

from src import config
from src.alert_system import AlertEngine
from src.live_inference import PredictiveMaintenanceService
from src.utils import get_logger

log = get_logger("api")

SERVICE = PredictiveMaintenanceService()
ENGINE = AlertEngine(verbose=False)
STARTED_AT = time.time()
REQUEST_LOG = deque(maxlen=200)
COUNTERS = {"requests": 0, "predictions": 0, "failures_flagged": 0, "errors": 0}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # ------------------------------------------------------------------ utils
    @app.before_request
    def _count():
        COUNTERS["requests"] += 1

    @app.after_request
    def _log(resp):
        REQUEST_LOG.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "path": request.path, "method": request.method, "status": resp.status_code,
        })
        return resp

    @app.errorhandler(Exception)
    def _boom(err):                                        # pragma: no cover
        COUNTERS["errors"] += 1
        log.exception("API error")
        return jsonify({"error": str(err), "type": type(err).__name__}), 400

    # --------------------------------------------------------------- endpoints
    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok" if SERVICE.ready else "model_not_loaded",
            "model_ready": SERVICE.ready,
            "uptime_seconds": round(time.time() - STARTED_AT, 1),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    @app.get("/model/info")
    def model_info():
        return jsonify(SERVICE.info())

    @app.post("/predict")
    def predict_ai4i():
        """AI4I-style machine snapshot -> predictive maintenance verdict."""
        payload = request.get_json(force=True, silent=True) or {}
        result = SERVICE.predict_ai4i(payload)
        COUNTERS["predictions"] += 1
        if result["will_fail"]:
            COUNTERS["failures_flagged"] += 1
        # register the alert so the plant gets a work order
        alert = ENGINE.process(payload.get("machine_id", "AI4I-UNIT"),
                               result["failure_probability"],
                               context={"source": "api", "verdict": result["verdict"]})
        if alert:
            result["work_order"] = {"id": alert.work_order_id,
                                    "severity": alert.severity,
                                    "action": alert.recommended_action}
        return jsonify(result)

    @app.post("/predict/window")
    def predict_window():
        """
        Body: {"machine_id": "MOTOR-012", "readings": [ {..}, {..}, ... ]}
        readings = the last N sensor samples, oldest first.
        """
        payload = request.get_json(force=True, silent=True) or {}
        readings = payload.get("readings") or payload.get("history") or []
        if not readings:
            return jsonify({"error": "provide a non-empty 'readings' list"}), 400

        import pandas as pd
        df = pd.DataFrame(readings)
        out = SERVICE.predict_stream(df, machine_id=payload.get("machine_id"))
        COUNTERS["predictions"] += 1
        alert = ENGINE.process(out["machine_id"], out["failure_probability"],
                               context={"source": "api/window"})
        if alert:
            out["work_order"] = {"id": alert.work_order_id, "severity": alert.severity}
        if out["will_fail"]:
            COUNTERS["failures_flagged"] += 1
        return jsonify(out)

    @app.post("/predict/reading")
    def predict_reading():
        """
        Edge style: push ONE reading; the service keeps a rolling buffer per
        machine and answers with the current risk - exactly how an ESP32 node
        would call it every 10 minutes.
        """
        payload = request.get_json(force=True, silent=True) or {}
        machine_id = payload.get("machine_id", "LIVE-001")
        reading = {k: v for k, v in payload.items() if k in config.STREAM_SENSOR_COLUMNS}
        if len(reading) < len(config.STREAM_SENSOR_COLUMNS):
            missing = set(config.STREAM_SENSOR_COLUMNS) - set(reading)
            return jsonify({"error": f"missing sensors: {sorted(missing)}"}), 400
        out = SERVICE.push(machine_id, reading)
        COUNTERS["predictions"] += 1
        alert = ENGINE.process(machine_id, out["failure_probability"],
                               context={"source": "api/reading"})
        if alert:
            out["work_order"] = {"id": alert.work_order_id, "severity": alert.severity,
                                 "action": alert.recommended_action}
        return jsonify(out)

    @app.get("/alerts")
    def alerts():
        return jsonify({"count": len(ENGINE.alerts),
                        "alerts": [a.to_row() for a in ENGINE.alerts[-50:]]})

    @app.get("/metrics")
    def metrics():
        return jsonify({**COUNTERS, "alerts_raised": len(ENGINE.alerts),
                        "recent_requests": list(REQUEST_LOG)[-10:],
                        "uptime_seconds": round(time.time() - STARTED_AT, 1)})

    # friendly index so opening the root in a browser is not a 404
    @app.get("/")
    def index():
        return jsonify({
            "service": "AI-Powered Predictive Maintenance API",
            "docs": "POST /predict  |  POST /predict/window  |  POST /predict/reading "
                    "|  GET /health  |  GET /model/info  |  GET /alerts  |  GET /metrics",
            "model_ready": SERVICE.ready,
        })

    return app


app = create_app()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    log.info("Starting PdM inference API on http://%s:%d", a.host, a.port)
    app.run(host=a.host, port=a.port, debug=a.debug)
