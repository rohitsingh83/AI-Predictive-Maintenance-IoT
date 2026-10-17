# 📘 PROJECT GUIDE — everything you need, in the exact order you need it

This single document is your build-and-ship handbook for
**"AI-Powered Predictive Maintenance System for IoT Devices"**.
It is written for a student building *proof of work*, and it contains the real
numbers this repository actually produced.

---

# A. PROJECT EXPLANATION

## A.1 In simple language

Machines — motors, pumps, compressors, gearboxes — rarely fail out of nowhere.
They *wear out* first: they get hotter, they start shaking more, they draw more
current, they sound different. Humans can't watch 30 machines at once, but sensors
can, and an AI can learn what "about to fail" looks like.

**This project reads six sensor signals from machines, learns the signature that
comes *before* a breakdown, and raises a maintenance work order about an hour
before the machine would have stopped the line.**

Three ways to maintain a machine:

| Strategy | When you act | Result |
|---|---|---|
| **Reactive** (run-to-failure) | after it breaks | maximum downtime, collateral damage |
| **Preventive** (calendar-based) | every N weeks regardless | you replace healthy parts, still get surprises |
| **Predictive** (this project) | when the data says it's needed | downtime ↓, parts used fully, surprises ↓ |

## A.2 The problem it solves

* Unplanned downtime is the single most expensive line item in manufacturing.
  One stopped machine on a line can cost thousands of dollars **per hour**.
* Emergency repairs cost 3–5× a planned repair, and collateral damage (a seized
  motor can destroy a gearbox) often exceeds the original part.
* Maintenance engineers drown in alarms and cannot triage 30 machines by eye.

Predictive maintenance turns "fix it when it breaks" into "replace this bearing
on Thursday, in the planned window" — same part, fraction of the cost.

## A.3 Why it matters per industry

| Industry | What it monitors | What it saves |
|---|---|---|
| **Manufacturing / factories** | motors, conveyors, CNC spindles, compressors | line throughput; avoids scrapped batches |
| **Power plants** | turbines, pumps, transformers, generators | grid stability; avoids forced outages |
| **Automotive** | stamping presses, robots, paint lines, welding guns | takt time; warranty cost |
| **Aviation / airlines** | engines, APUs, landing gear, avionics | AOG (aircraft on ground) costs ≈ $10k+/hour |
| **Oil & gas / utilities** | pumps, compressors, valves, pipelines | safety + environmental risk |
| **Data centres / facilities** | chillers, UPS, HVAC | SLA breaches |

**Companies doing this in production:** Siemens (Senseye), GE Vernova (APM),
Bosch, Schneider Electric, ABB, IBM Maximo, AWS IoT SiteWise, Azure IoT, Tesla.
**Public examples:** the U.S. Navy streams ~10,000 readings/second from the USS
*Fitzgerald* through AI models; Air France-KLM works with Google Cloud on
maintenance analytics; McDonald's uses AI to predict kitchen-equipment failures
across ~43,000 restaurants.

## A.4 In technical language

> A multivariate, imbalanced, non-stationary time-series classification problem
> over heterogeneous sensor channels. Each machine has a slow, latent degradation
> process; observable features (temperature, vibration, current, pressure, rpm,
> acoustic emission) drift non-linearly with a "knee" near end-of-life. The task
> is to estimate **P(failure within a horizon H | sensor history)** with
> **calibrated** probabilities, choose an operating point on the precision-recall
> curve that **minimises expected business cost**, and map the output into an
> actionable, de-duplicated maintenance workflow.

Key technical challenges and how this repo handles each:

| Challenge | Handling |
|---|---|
| Extreme class imbalance (2–3% positive) | class weights, PR-AUC as the primary metric, cost-tuned threshold |
| Non-stationarity (each machine is different) | per-machine baseline z-scores; rolling windows |
| Leakage risk | machine-wise split; explicit feature whitelist |
| Probability trustworthiness | isotonic calibration |
| Alarm fatigue | severity bands + de-duplication + escalation in the alert engine |
| Operating point | expected-cost minimisation, not 0.5 |

## A.5 The full workflow

```
1. SENSOR DATA COLLECTION   temp · vibration · current · pressure · rpm · acoustic
                            (10-min cadence in production; Edge gateway buffers)
2. PREPROCESSING            dedupe · physical bounds · interpolate drop-outs ·
                            machine-wise split (no leakage)
3. FEATURE ENGINEERING      physics features + rolling mean/std/min/max/range/trend
                            + deltas + per-machine baseline z-scores
4. MODEL TRAINING           4 learners benchmarked with 5-fold stratified CV,
                            winner calibrated (isotonic)
5. PREDICTION               P(failure within 6 cycles = 1 hour) per machine per cycle
6. DECISION                 threshold (cost-optimised) → LOW/MEDIUM/HIGH/CRITICAL
7. ALERT GENERATION         recommended action → work order, de-dup, webhook
8. VISUALISATION            operations dashboard, fleet health map, early-warning
                            timeline, business case
```

---

# B. TECH STACK OPTIONS

| | **Option A — Easiest** | **Option B — Intermediate (SELECTED ✅)** | **Option C — Advanced** |
|---|---|---|---|
| Language | Python | Python | Python |
| Libraries | pandas, NumPy, scikit-learn, matplotlib | + seaborn, joblib, Flask, pytest | + TensorFlow/PyTorch, SHAP, MLflow |
| Dataset | AI4I 2020 (tabular) | **AI4I 2020 (real) + virtual time-series fleet** | + private plant telemetry / larger MIMII / Paderborn bearing data |
| Model | Random Forest / Logistic Regression | **4-model CV benchmark (LR, RF, ET, HGB) + isotonic calibration** | LSTM/TCN/Transformer + RUL regression + survival analysis |
| Difficulty | ★☆☆☆☆ | ★★★☆☆ | ★★★★★ |
| GPU | not needed | not needed | strongly recommended |
| Runtime | < 1 min | 3–4 min full pipeline | hours |
| Output | accuracy + a plot | metrics, dashboard, REST API, work orders, business case | + deployed pipeline, cloud, drift monitoring |
| Best for | "I need something working today" | **placements: best effort-to-impact ratio** | research / ML-heavy roles |

## Selected: **Option B** — and why

1. **Runs on any laptop** (no GPU, no cloud, no money) — you'll actually finish it.
2. **Real dataset + virtual simulation** answers the "you don't have hardware"
   objection *and* the "tabular data isn't real IoT" objection in one project.
3. **Every skill recruiters screen for:** pandas/NumPy, scikit-learn, feature
   engineering, imbalanced metrics, calibration, deployment (Flask), testing (pytest),
   CI (GitHub Actions), Docker, documentation.
4. **Two distinct ML problems** (tabular classification + time-series early warning)
   with honest evaluation, which is exactly what separates a portfolio from a tutorial.

---

# C. PROJECT ARCHITECTURE

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full block diagram. Summary:

```
Sensors ─▶ Edge gateway ─▶ Cleaning ─▶ Feature engineering ─▶ Model
       ─▶ Probability ─▶ Risk band ─▶ Work order ─▶ Dashboard / CMMS / actuation
```

**Module responsibilities**

| Module | Does what | Why it matters |
|---|---|---|
| `src/config.py` | all paths, columns, thresholds, cost constants | one place to change everything |
| `src/simulate_iot_sensors.py` | 30-machine virtual fleet with hidden health index | replaces missing hardware, honestly |
| `src/data_loader.py` | loads real AI4I + virtual stream (download/fallback) | never breaks without internet |
| `src/preprocessing.py` | clean, impute, bound, **machine-wise split** | leak-free evaluation |
| `src/feature_engineering.py` | physics + window + baseline features | the biggest performance lever |
| `src/model_training.py` | 4-model CV benchmark, calibration, threshold tuning | evidence-based model choice |
| `src/evaluation.py` | PR/ROC-AUC, MCC, Brier, **cost model** | metrics that mean something |
| `src/alert_system.py` | severity bands, de-dup, work orders | turns prediction into action |
| `src/business_impact.py` | savings, scenarios, payback | the money slide |
| `src/visualization.py` | 17 figures | makes the repo look like a product |
| `src/live_inference.py` | rolling buffer → features → verdict | model → live service |
| `src/dashboard.py` | self-contained HTML console | demo anyone can open |
| `api/app.py` | Flask REST API | integration surface |
| `realtime_simulator.py` | pushes "live factory" readings | demo without hardware |
| `hardware/raspberry_pi_sensor_node.py` | real sensor node | optional physical rig |

**Data flow (concrete)**

`iot_sensor_stream.csv` (27,000 rows × 30 machines)
→ clean → 119 columns after feature engineering → 92 model inputs
→ ExtraTrees → `P(failure within 6 cycles)` → threshold 0.36
→ severity → `WO-1xxx` → `outputs/live_alerts.csv` → dashboard & API.

---

# D. FOLDER STRUCTURE (and why each folder exists)

```
AI-Predictive-Maintenance-IoT/
├── api/app.py          # REST API - the integration surface for SCADA/CMMS
├── data/raw/           # AI4I real dataset (committed) + generated stream
├── data/processed/     # cleaned/split files (regenerated, git-ignored)
├── docs/               # ARCHITECTURE, MODEL_CARD, DEPLOYMENT, INTERVIEW_GUIDE, this file
├── hardware/           # Raspberry Pi node + BOM (optional physical rig)
├── images/             # 17 generated figures - the visual proof in the README
├── models/             # model bundles (.pkl, ~18 MB each) + metrics + model cards
├── notebooks/          # 3 teaching notebooks: EDA, FE, training/eval
├── outputs/            # dashboard.html, alerts, predictions, business impact json
├── reports/            # business_impact.md, model comparison CSVs
├── src/                # all reusable, importable logic (14 modules)
├── tests/              # 10 pytest tests
├── .github/workflows/  # CI: install → simulate → train → test → build
├── main.py             # single entry point: simulate/train/evaluate/visualize/report/dashboard
├── train_model.py      # training CLI
├── realtime_simulator.py # fake live factory feeding the API
├── Dockerfile, docker-compose.yml, Makefile
├── requirements.txt, .gitignore, LICENSE, README.md
```

Rules of thumb used here (worth copying into every project):

* `src/` = logic, importable, no side effects. Entry-point scripts = thin wrappers.
* Anything **generated** is git-ignored (`data/processed`, `outputs/*.log`, `images/*`? no —
  figures ARE committed because the README embeds them).
* Anything **hand-written and important** is committed: code, docs, figures, small public data.
* Configuration never lives inside logic (`src/config.py`).

---

# E. INSTALLATION

**Python version:** 3.9+ (developed and verified on 3.13; CI tests 3.10 & 3.11).

## Windows

```powershell
git clone https://github.com/rohitsingh83/AI-Predictive-Maintenance-IoT.git
cd AI-Predictive-Maintenance-IoT
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py all
```

## macOS / Linux

```bash
git clone https://github.com/rohitsingh83/AI-Predictive-Maintenance-IoT.git
cd AI-Predictive-Maintenance-IoT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py all
```

**If something fails:**
* `python: command not found` → use `python3`, or install Python from python.org.
* `ModuleNotFoundError: src` → run commands **from the repo root** (not from `src/`).
* Matplotlib GUI error on a server → figures are already saved to `images/`.
* Slow first run → the AI4I dataset downloads once (~500 KB) into `data/raw/`.

---

# F. THE CODE (what each file contains)

| File | Lines | What you learn from it |
|---|---|---|
| `src/config.py` | ~120 | centralised configuration, risk bands, cost constants |
| `src/simulate_iot_sensors.py` | ~200 | physics-based simulation, hidden ground truth, labelling horizons |
| `src/data_loader.py` | ~145 | real-dataset loading, download with graceful fallback |
| `src/preprocessing.py` | ~180 | cleaning rules, physical bounds, **machine-wise splitting** |
| `src/feature_engineering.py` | ~180 | physics features, rolling windows, baseline z-scores, leakage guards |
| `src/model_training.py` | ~230 | model zoo, cross-validation, calibration, artefact bundling |
| `src/evaluation.py` | ~150 | imbalanced metrics, cost-based threshold tuning, business impact |
| `src/alert_system.py` | ~250 | severity banding, de-duplication, work orders, webhook, lead-time analysis |
| `src/business_impact.py` | ~230 | scenario modelling, honest financial assumptions |
| `src/visualization.py` | ~400 | 17 publication-quality figures with house style |
| `src/live_inference.py` | ~250 | rolling buffers, feature reconstruction, explanations |
| `src/dashboard.py` | ~330 | self-contained HTML + embedded base64 images + JS replay |
| `api/app.py` | ~230 | REST endpoints, request validation, graceful errors |
| `main.py` | ~240 | argparse CLI, orchestration, model cards |

Read them in this order for maximum learning:
`config → simulate → preprocessing → feature_engineering → model_training → evaluation → alert_system → api → dashboard`.

---

# G. VIRTUAL SIMULATION (your substitute for hardware)

## G.1 Why it's legitimate

A simulation is a **generative model of the physics you cannot access**. It is
standard industrial practice (digital twins) — and unlike a random-data demo, this
one encodes real degradation behaviour, so the ML problem it creates is realistic
and the results are meaningful.

## G.2 How the dataset simulates real IoT sensors

Each machine has a **hidden health index h ∈ [0,1]** (ground truth; the model never
sees it):
* `h` decays slowly, then **accelerates past a "knee"** (real bearing degradation).
* Sensors react **non-linearly and coupled**:
  * temperature ↑ with degradation + ambient + shift load
  * vibration ↑ **exponentially** (`v ∝ e^(2.35·damage)`)
  * current ↑ (friction and efficiency loss)
  * pressure ↓ (internal leakage)
  * rpm ↓ slightly under load
  * acoustic emission ↑ sharply near end-of-life
* Four failure modes with different sensor gains: `bearing_wear`, `overheating`,
  `electrical_fault`, `lubrication_loss`.
* Real-world noise: sensor glitches (0.4% NaNs), shift load cycles, ambient swing,
  planned services that reset health, emergency repairs after breakdowns.

## G.3 How failure is represented

| Column | Meaning | Used for |
|---|---|---|
| `health_index` | hidden true condition | ground truth only (never a model input) |
| `failure_now` | the machine breaks on this cycle | detection task |
| `failure_within_horizon` | breaks within the next 6 cycles | **the predictive target** |
| `cycles_to_failure` | countdown | analysis |
| `fault_mode` | which failure mode | alert messaging |

## G.4 Step-by-step simulation walkthrough

```
Step 1  python main.py simulate
        → data/raw/iot_sensor_stream.csv   (27,000 rows, 30 machines, 118 breakdowns)

Step 2  python main.py train --dataset stream
        → 92 window features per row
        → machine-wise split: 24 machines train / 6 machines test (never seen)
        → 4 models benchmarked → ExtraTrees wins (PR-AUC 0.995)
        → models/stream_failure_model.pkl  +  models/stream_metrics.json

Step 3  python -m src.alert_system
        → replays the held-out predictions through the alert engine
        → outputs/live_alerts.csv  (103 work orders)
        → outputs/alert_lead_time_report.csv
        → 100% of 39 breakdowns flagged; average 5.3 cycles (0.9 h) notice

Step 4  python main.py visualize
        → images/09_degradation_trajectory.png   (a machine wearing out)
        → images/10_early_warning_timeline.png   (alarm before the breakdown)
        → images/11_fleet_health_map.png         (which machine to service first)

Step 5  python main.py dashboard
        → outputs/dashboard.html  (press "Play replay")

Step 6  python realtime_simulator.py --local     # optional live demo
```

## G.5 Outputs to generate, graphs to show, proof to capture

| Evidence | File | Why it convinces |
|---|---|---|
| Sensor degradation | `images/09_degradation_trajectory.png` | shows the physics is real |
| Early warning | `images/10_early_warning_timeline.png` | the headline: alarm before failure |
| Fleet triage | `images/11_fleet_health_map.png` | shows operational thinking |
| Metrics | `images/04_confusion_matrix.png`, `13_..._stream.png` | honest evaluation |
| Threshold economics | `images/07_threshold_optimisation.png` | business maturity |
| Live console | `outputs/dashboard.html` | it *looks* shipped |
| Work orders | `outputs/live_alerts.csv` | prediction → action |
| Money | `reports/business_impact.md` | ROI thinking |

---

# H. HOW TO RUN THE PROJECT

```bash
python main.py all            # simulate → train → evaluate → alerts → figures → report → dashboard
python main.py demo           # 30-second live console demo (record this!)
python main.py predict --json '{"air_temp_k":302,"process_temp_k":312,
                                "rotational_speed_rpm":1250,"torque_nm":68,
                                "tool_wear_min":240}'
python api/app.py             # API on http://localhost:5000
python realtime_simulator.py  # live factory → API
pytest -q                     # 10 tests
```

## Expected results (this repo, seed = 42)

| Metric | AI4I (real data) | Virtual fleet (unseen machines) |
|---|---|---|
| Model | RandomForest | ExtraTrees |
| Threshold | 0.26 | 0.36 |
| ROC-AUC | 0.9782 | 0.9998 |
| PR-AUC | 0.8706 | 0.9951 |
| Recall | 0.8088 | 0.9701 |
| Precision | 0.8730 | 0.9458 |
| F1 | 0.8397 | 0.9578 |
| Confusion | TP 55 · FP 8 · FN 13 · TN 1924 | TP 227 · FP 13 · FN 7 · TN 5153 |
| Early warning | — | 100% of 39 breakdowns, avg 0.9 h notice |
| Business | — | $1,193,012/yr saved (expected), payback 2.3 days |

**Sample console output**

```
ALERT WO-1103 | CRITICAL | MOTOR-011 | P(failure)=0.92 | STOP & inspect now - schedule emergency maintenance

--- ALERT EFFECTIVENESS (held-out machines) ---
 machine_id  breakdown_cycle  alerts_before_breakdown  lead_time_cycles  lead_time_hours  predicted_in_advance
 MOTOR-019              839                        3                7             1.17                  True

Breakdowns predicted in advance : 100% (39/39)
Average early-warning lead time : 5.3 cycles (0.9 h of advance notice)
```

---

# I. GITHUB UPLOAD STRATEGY

## I.1 Repository identity

* **Name:** `AI-Predictive-Maintenance-IoT`
  (alternatives: `predictive-maintenance-mlops`, `iot-predictive-maintenance-ai`)
* **Description (about):**
  > End-to-end AI predictive maintenance for IoT-instrumented machines: sensor
  > simulation, physics + time-series feature engineering, cost-optimised ML
  > (0.978 ROC-AUC), Flask API, alerting and an HTML operations dashboard.
* **Topics/tags:**
  `predictive-maintenance` `iot` `machine-learning` `scikit-learn` `anomaly-detection`
  `time-series` `industry40` `smart-manufacturing` `flask-api` `data-science`
  `python` `condition-monitoring` `edge-computing` `mlops` `student-project`
* **Website field:** link to `outputs/dashboard.html` on GitHub Pages (Settings → Pages →
  deploy from branch, folder `/`), so recruiters can click and see the dashboard live.

## I.2 First push

```bash
cd AI-Predictive-Maintenance-IoT
git init
git branch -M main
git add .
git status                      # verify: no .venv, no __pycache__, no huge files
git commit -m "feat: complete AI predictive maintenance system (models, API, dashboard, docs)"
git remote add origin https://github.com/rohitsingh83/AI-Predictive-Maintenance-IoT.git
git push -u origin main
```

## I.3 Commit strategy (conventional commits, small and honest)

```
feat: add virtual IoT sensor simulator with degradation physics
feat(data): load AI4I 2020 dataset with download fallback
feat(features): add physics-based engineered features
feat(features): add rolling-window and baseline time-series features
feat(model): benchmark 4 classifiers with stratified CV
feat(model): calibrate probabilities and tune threshold by business cost
feat(alerts): add severity bands, de-duplication and work orders
feat(api): expose /predict, /predict/window, /predict/reading
feat(dashboard): build self-contained HTML operations console
docs: add architecture, model card, deployment and interview guides
test: cover physics invariants, leakage guards, metrics and API routes
ci: run simulate → train → pytest → build on push
build: add Dockerfile, compose and Makefile
```

**Rules:** one logical change per commit · imperative mood · never "update" or "fix stuff" ·
commit the regenerated figures with the code that made them.

## I.4 Making the repo look professional

1. README first screen: title, badges, one-line pitch, hero figure, quick start.
2. All 17 figures embedded in the README (recruiters rarely open folders).
3. `docs/` with architecture, model card, deployment and interview guide.
4. A **model card** — almost no student repo has one; it signals maturity.
5. CI badge green.
6. Pin the repo on your profile; add the topic tags.
7. Add a "Results" table with numbers a reviewer can re-verify with one command.
8. License + citation block.
9. Keep the commit history clean (no 40 commits called "final").
10. Add a 60-second screen recording (Loom/GIF) of `python main.py demo` and the
    dashboard replay to the README — the single highest-impact addition.

---

# J. README CONTENT (already written for you)

`README.md` in this repo contains, in order:

1. Badges + one-line pitch
2. Overview table (task, dataset, models, results, stack)
3. Problem statement (with the concrete 30-machine scenario)
4. Industry relevance (companies, savings ranges, named public case studies)
5. Architecture (mermaid diagram + expandable layer explanation)
6. Folder structure (annotated)
7. Quick start + all CLI commands
8. API usage with real request/response JSON
9. Results (model comparison table, confusion matrix, early-warning table, key figures)
10. Business impact (scenario table)
11. Testing & CI
12. "What makes this different" comparison table
13. Learning outcomes
14. Limitations (honesty = credibility)
15. Roadmap (shows you think beyond the assignment)
16. References, license, citation, roadmap
17. Resume bullets (in `docs/INTERVIEW_GUIDE.md`)

---

# K. STEP-BY-STEP 7-DAY PROOF PLAN

| Day | Goal | Do this | Commit | Proof to capture |
|---|---|---|---|---|
| **1** | Setup + skeleton | create venv, folders, `src/config.py`, `requirements.txt`, `.gitignore`, empty `main.py` | `chore: project skeleton and configuration` | screenshot of folder tree + `python -V` in venv |
| **2** | Data | `src/data_loader.py`, `src/simulate_iot_sensors.py`, run `python main.py simulate` | `feat: virtual IoT fleet simulator + AI4I loader` | `data/raw/*` head screenshot, sensor describe() table, `images/09_...png` |
| **3** | Preprocessing + features | `preprocessing.py`, `feature_engineering.py`, show leakage guards | `feat: preprocessing with leak-free machine-wise split` / `feat: physics + window feature engineering` | feature-count output, correlation heatmap `02_...png` |
| **4** | Model | `model_training.py`, run `python main.py train` | `feat: 4-model CV benchmark with calibration and cost-tuned threshold` | training log screenshot, `03_...png`, `04_...png` |
| **5** | Evaluation + business | `evaluation.py`, `business_impact.py`, run `python main.py evaluate report` | `feat: imbalanced metrics and business impact model` | `05/06/07/17_*.png`, `reports/business_impact.md` |
| **6** | Serving + visuals | `alert_system.py`, `live_inference.py`, `api/app.py`, `dashboard.py`, `visualization.py` | `feat: alert engine, REST API and HTML dashboard` | curl output, `outputs/dashboard.html` screenshot, `10_...png` |
| **7** | Docs + polish + publish | README, docs, tests, CI, Dockerfile; `pytest -q`; push | `docs: add guides` / `test: add suite` / `ci: add workflow` | green CI badge, repo preview screenshot, final figure wall |

**Daily proof ritual (5 minutes):** run the command → screenshot the terminal →
save into `images/proof/` → tweet/LinkedIn the result with one sentence about what
you learned. Seven days of that is a portfolio, not just a repo.

---

# L. PROOF CHECKLIST / SCREENSHOTS TO CAPTURE

| # | Screenshot | Source | Where it goes |
|---|---|---|---|
| 1 | Dataset preview (head + shape) | `raw.head()` / `data/raw/*.csv` | README + Day-2 post |
| 2 | Sensor summary table | `python main.py simulate` output | README |
| 3 | Preprocessing output (before/after NA counts) | notebook 02 | docs |
| 4 | Model training log (CV comparison table) | `python main.py train` output | README "Results" |
| 5 | Confusion matrix | `images/04_...png`, `13_...png` | README |
| 6 | ROC + PR curves | `images/05_...png`, `15_...png` | README |
| 7 | Feature importance | `images/06_...png`, `16_...png` | README |
| 8 | Threshold/cost curve | `images/07_...png` | README |
| 9 | Degradation trajectory | `images/09_...png` | README |
| 10 | **Early-warning timeline** | `images/10_...png` | hero image |
| 11 | Fleet health map | `images/11_...png` | README |
| 12 | Business impact charts | `images/17_...png` | README + LinkedIn |
| 13 | Live dashboard (Play replay mid-animation) | `outputs/dashboard.html` | README + demo video |
| 14 | API request/response (curl or Postman) | `curl -X POST /predict` | README API section |
| 15 | Alert log / work orders | `outputs/live_alerts.csv` | README |
| 16 | Test suite passing | `pytest -q` output | README |
| 17 | Green CI run | GitHub Actions tab | README badge |
| 18 | Repo preview (README rendered) | GitHub page | LinkedIn/portfolio |

**File naming convention:** `NN_topic.png` (e.g. `10_early_warning_timeline.png`) —
numbered so they appear in narrative order in the folder and README.
**Storage:** `images/` (committed), raw screenshots in `images/proof/` (optional, git-ignored
if large). **README usage:** embed with `![caption](images/NN_topic.png)` and always add
one sentence explaining *what the reader should notice*.

---

# M. FINAL CHECKLIST BEFORE YOU PUBLISH

- [ ] `pytest -q` passes (10 tests)
- [ ] `python main.py all` runs clean on a fresh clone
- [ ] README renders on GitHub with all figures visible
- [ ] No secrets, no `.venv`, no `__pycache__` committed (`.gitignore` covers it)
- [ ] Repo description + topics set
- [ ] License present (MIT) + dataset attributed (AI4I, CC BY 4.0)
- [ ] Model card committed
- [ ] CI green
- [ ] Demo video recorded (dashboard replay + `main.py demo`)
- [ ] Resume bullets updated with the real numbers
- [ ] LinkedIn/X post drafted with the early-warning figure
- [ ] Pinned on your GitHub profile

## The 20-second elevator version (for interviews)

> "I built a predictive maintenance system for IoT-instrumented machines. It reads six
> sensor channels, engineers physics and rolling-window features, and predicts failures
> before they happen — 0.978 ROC-AUC on the real AI4I industrial dataset and 0.958 F1 on
> a 30-machine virtual fleet where the model had never seen the test machines, with
> about an hour of advance warning. I tuned the alarm threshold by business cost rather
> than accuracy, so a missed failure costs ~16× a false alarm, and I shipped it as a
> Flask API with a dashboard, work-order alerting and a full test suite."
