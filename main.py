"""
main.py
=======
ONE entry point for the whole project - the file a recruiter or reviewer runs.

    python main.py simulate      # 1. generate the virtual IoT sensor fleet
    python main.py train         # 2. train both AI models (AI4I + time-series)
    python main.py evaluate      # 3. metrics, business case, model card
    python main.py visualize     # 4. build every figure
    python main.py report        # 5. business impact report (markdown + json)
    python main.py dashboard     # 6. build outputs/dashboard.html
    python main.py predict --json '{"air_temp_k":302,"process_temp_k":312,
                                    "rotational_speed_rpm":1250,"torque_nm":68,
                                    "tool_wear_min":240}'
    python main.py all           # everything, in the right order (≈3-4 min)
    python main.py demo          # cute 20-second console demo for a screen recording

Every sub-command is also importable, so the notebooks can reuse them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.utils import banner, get_logger, json_safe, print_table

log = get_logger("main")


# ---------------------------------------------------------------------------
def cmd_simulate(args):
    from src.simulate_iot_sensors import simulate_fleet
    banner("STEP 1/6 - VIRTUAL IoT FACTORY SIMULATION")
    df = simulate_fleet(n_machines=args.machines, n_cycles=args.cycles)
    print(f"\nGenerated {len(df):,} sensor samples from {df.machine_id.nunique()} machines "
          f"({int(df.failure_now.sum())} real breakdowns)")
    print(df[config.STREAM_SENSOR_COLUMNS].describe().T[["mean", "std", "min", "max"]].round(2))
    return df


def cmd_train(args):
    banner("STEP 2/6 - MODEL TRAINING")
    from train_model import train_ai4i, train_stream
    if args.dataset in ("ai4i", "both"):
        train_ai4i(args.fast, args.threshold_mode)
    if args.dataset in ("stream", "both"):
        train_stream(args.fast, args.threshold_mode)


def cmd_evaluate(args):
    banner("STEP 3/6 - EVALUATION")
    from src.model_training import feature_importance
    import joblib
    for name, path, metrics in [("AI4I-2020 (real factory data)", config.BEST_MODEL_PATH,
                                 config.AI4I_METRICS),
                                ("Virtual IoT stream", config.STREAM_MODEL_PATH,
                                 config.STREAM_METRICS)]:
        if not metrics.exists():
            print(f"[skip] {name}: no metrics yet - run `python main.py train`")
            continue
        meta = json.loads(metrics.read_text())
        m = meta["test_metrics"]
        print(f"\n=== {name} ===  best model: {meta['chosen_model']}")
        print(f"  ROC-AUC {m['roc_auc']:.4f} | PR-AUC {m['pr_auc']:.4f} | "
              f"Recall {m['recall']:.4f} | Precision {m['precision']:.4f} | F1 {m['f1']:.4f}")
        print(f"  confusion: TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} "
              f"| threshold {m['threshold']}")
        bundle = joblib.load(path)
        imp = feature_importance(bundle, 8)
        if not imp.empty:
            print("  top drivers:")
            print(imp[["feature", "importance_pct"]].to_string(index=False))
        # model card
        card = {"dataset": name, "model": meta["chosen_model"],
                "features": meta["n_features"], "threshold": m["threshold"],
                "metrics": m, "top_features": imp.to_dict(orient="records"),
                "trained_at": bundle["trained_at"], "seed": config.RANDOM_SEED,
                "limitations": [
                    "Trained on a public benchmark (AI4I 2020) and a virtual fleet; "
                    "real plants must re-train on their own historical work orders.",
                    "Predicts failure within the configured horizon only - it is not a "
                    "remaining-useful-life (RUL) regressor.",
                    "Sensor drift / recalibration can silently degrade accuracy - monitor "
                    "input distributions in production (data-drift checks).",
                ]}
        print(f"  model card -> models/model_card_{'ai4i' if 'AI4I' in name else 'stream'}.json")
        from src.utils import save_json
        save_json(card, config.MODELS_DIR /
                  f"model_card_{'ai4i' if 'AI4I' in name else 'stream'}.json")


def cmd_visualize(args):
    banner("STEP 4/6 - FIGURE GENERATION")
    from src.visualization import generate_all_figures
    files = generate_all_figures()
    print(f"\n{len(files)} figures written to {config.IMAGES_DIR}/")


def cmd_report(args):
    banner("STEP 5/6 - BUSINESS IMPACT REPORT")
    from src.business_impact import build_report, write_markdown
    rep = build_report()
    write_markdown(rep)
    print(json.dumps(json_safe(rep["headline_expected"]), indent=2))
    print(f"\nFull report -> {config.BUSINESS_REPORT}")


def cmd_dashboard(args):
    banner("STEP 6/6 - DASHBOARD")
    from src.dashboard import build_dashboard
    p = build_dashboard()
    print(f"\nOpen in a browser: {p}")


def cmd_predict(args):
    banner("SINGLE PREDICTION")
    from src.live_inference import PredictiveMaintenanceService
    svc = PredictiveMaintenanceService()
    if args.json:
        payload = json.loads(args.json)
    else:                                   # realistic demo default: worn tool + high torque
        payload = {"air_temp_k": 302.0, "process_temp_k": 312.0,
                   "rotational_speed_rpm": 1250, "torque_nm": 68.0, "tool_wear_min": 240}
    print("input:", json.dumps(payload))
    print(json.dumps(svc.predict_ai4i(payload), indent=2))


def cmd_all(args):
    args.fast = args.fast; args.dataset = "both"; args.threshold_mode = config.THRESHOLD_MODE
    cmd_simulate(argparse.Namespace(machines=30, cycles=900))
    cmd_train(args)
    cmd_evaluate(args)
    then_alerts()
    cmd_visualize(args)
    cmd_report(args)
    cmd_dashboard(args)
    banner("DONE")
    print("Artefacts: models/ | images/ | outputs/ | reports/")
    print("Next: python api/app.py   (REST API)   |   open outputs/dashboard.html")


def then_alerts():
    banner("STEP 3b - ALERT ENGINE REPLAY")
    from src.alert_system import replay_history
    df = replay_history(verbose=False)
    print(f"{len(df)} alerts raised and logged -> {config.LIVE_ALERTS_CSV}")


def cmd_demo(args):
    """Short scripted demo: nice for a 30-second screen recording."""
    import numpy as np
    import pandas as pd
    import time as _t
    from src.live_inference import PredictiveMaintenanceService
    from src.alert_system import AlertEngine

    banner("LIVE DEMO - IoT EDGE FEED -> AI -> WORK ORDER")
    svc = PredictiveMaintenanceService()
    engine = AlertEngine(verbose=False)
    stream = pd.read_csv(config.STREAM_RAW_CSV)
    mid = stream.loc[stream.failure_now == 1, "machine_id"].iloc[0]
    fail_cycle = int(stream.loc[(stream.machine_id == mid) & (stream.failure_now == 1),
                                "cycle"].iloc[0])
    d = stream[(stream.machine_id == mid) &
               stream.cycle.between(fail_cycle - 40, fail_cycle + 1)].sort_values("cycle")

    history = []
    for _, row in d.iterrows():
        history.append({c: row[c] for c in config.STREAM_SENSOR_COLUMNS})
        verdict = svc.predict_stream(pd.DataFrame(history[-24:]), machine_id=mid)
        p = verdict["failure_probability"]
        bar = "#" * int(p * 30)
        print(f"cycle {int(row.cycle):>4} | temp {row.temperature_c:6.1f} degC | "
              f"vib {row.vibration_mm_s:6.2f} mm/s | P(fail)={p:0.3f} |{bar:<30}| "
              f"{verdict['severity']}")
        if p >= 0.30:
            engine.process(mid, p, context={"cycle": int(row.cycle)})
        _t.sleep(max(0.0, args.delay))
    print("\nDemo finished - note the work order(s) raised BEFORE the breakdown.")


# ---------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="AI-Powered Predictive Maintenance System for IoT Devices - "
                    "end-to-end pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="command")

    p = sub.add_parser("simulate", help="generate the virtual IoT sensor fleet")
    p.add_argument("--machines", type=int, default=30)
    p.add_argument("--cycles", type=int, default=900)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("train", help="train the AI models")
    p.add_argument("--dataset", choices=["ai4i", "stream", "both"], default="both")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--threshold-mode", choices=["cost", "f1"], default=config.THRESHOLD_MODE)
    p.set_defaults(func=cmd_train)

    for name, fn, helptext in [("evaluate", cmd_evaluate, "show metrics + model cards"),
                               ("visualize", cmd_visualize, "generate all figures"),
                               ("report", cmd_report, "business impact report"),
                               ("dashboard", cmd_dashboard, "build outputs/dashboard.html")]:
        sp = sub.add_parser(name, help=helptext)
        sp.set_defaults(func=fn)

    p = sub.add_parser("predict", help="score one machine snapshot")
    p.add_argument("--json", type=str, default=None,
                   help='e.g. \'{"air_temp_k":302,"process_temp_k":312,"rotational_speed_rpm":1250,'
                        '"torque_nm":68,"tool_wear_min":240}\'')
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("all", help="run the complete pipeline end-to-end")
    p.add_argument("--fast", action="store_true")
    p.set_defaults(func=cmd_all)

    p = sub.add_parser("demo", help="30-second live console demo (great for recordings)")
    p.add_argument("--delay", type=float, default=0.08)
    p.set_defaults(func=cmd_demo)

    return ap


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        sys.exit(0)
    args.func(args)
