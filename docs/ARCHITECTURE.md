# System Architecture

## 1. Text block diagram

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 - PHYSICAL / VIRTUAL MACHINE + IoT SENSORS                               │
│                                                                                   │
│   Rotating machine (motor / pump / compressor / CNC spindle)                      │
│        ├── DHT22       → temperature (°C)          [overheating]                   │
│        ├── SW-420/MEMS → vibration (mm/s RMS)      [bearing wear, imbalance]       │
│        ├── ACS712+ADC  → current (A)               [electrical fault, overload]    │
│        ├── pressure Tx → pressure (bar)            [leakage, blockage]             │
│        ├── tachometer  → rpm                        [load, slip]                    │
│        └── acoustic    → dB                         [friction, cavitation]          │
│   Sampling: every 10 minutes (production) / 1-10 s (demo)                          │
└───────────────────────────────┬───────────────────────────────────────────────────┘
                                │  MQTT / HTTPS / Modbus-TCP
┌───────────────────────────────▼───────────────────────────────────────────────────┐
│  LAYER 2 - EDGE GATEWAY  (Raspberry Pi / ESP32 / industrial IPC)                   │
│   • ingest + timestamp + buffer readings (rolling window per machine)              │
│   • validate ranges, drop sensor drop-outs                                        │
│   • run the light model locally (offline capability) OR forward to cloud           │
│   • relay actuation: stop motor / sound alarm / open bypass on CRITICAL            │
└───────────────────────────────┬───────────────────────────────────────────────────┘
                                │  batched telemetry
┌───────────────────────────────▼───────────────────────────────────────────────────┐
│  LAYER 3 - DATA / FEATURE PIPELINE            src/preprocessing.py                 │
│                                               src/feature_engineering.py           │
│   clean → impute → physical bounds → split (machine-wise, leak-free)               │
│   features: raw sensors + physics features + rolling mean/std/min/max/range/trend  │
│            + deltas + per-machine baseline z-scores  (92 model inputs)             │
└───────────────────────────────┬───────────────────────────────────────────────────┘
                                │  feature matrix
┌───────────────────────────────▼───────────────────────────────────────────────────┐
│  LAYER 4 - AI MODELS                          src/model_training.py                │
│                                                                                    │
│   Model A: AI4I 2020 (real factory data)                                           │
│     LogisticRegression | RandomForest | ExtraTrees | HistGradientBoosting          │
│     → 5-fold stratified CV → cost-tuned threshold → isotonic calibration           │
│     chosen: RandomForest   ROC-AUC 0.978 | PR-AUC 0.871 | F1 0.840                 │
│                                                                                    │
│   Model B: virtual IoT time-series early warning                                   │
│     same 4 learners on 92 window features, machine-wise split                       │
│     chosen: ExtraTrees     ROC-AUC 0.9998 | PR-AUC 0.995 | F1 0.958                │
└───────────────────────────────┬───────────────────────────────────────────────────┘
                                │  P(failure within 6 cycles) + severity band
┌───────────────────────────────▼───────────────────────────────────────────────────┐
│  LAYER 5 - DECISION & ALERTING                src/alert_system.py                  │
│   LOW / MEDIUM / HIGH / CRITICAL  → recommended action → work order (WO-xxxx)      │
│   de-duplication + escalation (no alarm fatigue) + webhook (Slack/Teams/e-mail)    │
│   measured: 100% of held-out breakdowns flagged, 0.9 h average advance notice      │
└───────────────────────────────┬───────────────────────────────────────────────────┘
                                │  downstream
┌───────────────────────────────▼───────────────────────────────────────────────────┐
│  LAYER 6 - SERVING & VISUALISATION                                                 │
│   • api/app.py         Flask REST API  (/predict, /predict/window, /alerts)        │
│   • outputs/dashboard.html  operator console (KPI cards, live replay, figures)     │
│   • reports/business_impact.md  finance-ready business case                        │
│   • outputs/*.csv      audit trail: predictions, alerts, lead-time report          │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Data flow in one line

```
sensors ─▶ edge buffer ─▶ cleaning ─▶ feature engineering ─▶ model ─▶ probability
        ─▶ risk band ─▶ work order ─▶ dashboard / CMMS / actuation
```

## 3. Module map

| File | Responsibility | Key idea |
|---|---|---|
| `src/config.py` | paths, columns, thresholds, cost constants | one place to change everything |
| `src/simulate_iot_sensors.py` | virtual factory (30 machines, hidden health index) | replaces missing hardware honestly |
| `src/data_loader.py` | load real AI4I + the virtual stream, with download/fallback | never crash without internet |
| `src/preprocessing.py` | clean, impute, bound, split (machine-wise) | leak-free evaluation |
| `src/feature_engineering.py` | physics + window + baseline features | domain knowledge beats raw sensors |
| `src/model_training.py` | 4-model CV benchmark, calibration, threshold tuning | evidence-based model choice |
| `src/evaluation.py` | PR/ROC-AUC, MCC, Brier, confusion, cost model | metrics a plant manager understands |
| `src/alert_system.py` | severity bands, de-duplication, work orders | predictions → action |
| `src/business_impact.py` | annual savings, payback, scenarios | the money slide |
| `src/visualization.py` | 16 publication-quality figures | repo looks like a product |
| `src/live_inference.py` | rolling buffer, feature rebuild, verdicts | model → live service |
| `src/dashboard.py` | self-contained HTML console | demo anyone can open |
| `api/app.py` | REST API | integration surface |
| `realtime_simulator.py` | pushes live readings into the API | demo without hardware |
| `hardware/raspberry_pi_sensor_node.py` | real sensor node, offline-capable | optional physical rig |

## 4. Design decisions worth defending in an interview

1. **Two datasets, one pipeline.** AI4I gives *real* industrial credibility; the virtual
   fleet gives *time-series* early-warning capability. Neither alone tells the whole story.
2. **Machine-wise split, not random.** Random row splitting lets the model memorise
   machine personality. Splitting by machine answers: *does it work on a new machine?*
3. **Cost-tuned threshold, not 0.5.** A missed failure is ~16x more expensive than a
   false alarm, so the operating point is a business decision, not a default.
4. **Isotonic calibration.** Risk bands (MEDIUM/HIGH/CRITICAL) are only meaningful if
   `P=0.80` really means ~80% of such machines fail.
5. **No leakage by construction.** `stream_feature_columns()` filters ground-truth
   columns (`health_index`, `failure_*`) and per-machine baseline constants.
6. **Simulator is honest.** Every failure rate, sensor gain and degradation shape is
   documented in `simulate_iot_sensors.py` - a reviewer can audit the physics.
