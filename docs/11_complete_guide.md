# 11. Complete Master Guide — AI Predictive Maintenance for IoT

---

## 11.1 Executive Summary

This master guide consolidates the engineering principles, data pipelines, machine learning architecture, edge deployment specs, and business impact modeling of the **AI-Powered Predictive Maintenance System for IoT Devices**.

---

## 11.2 Core System Specifications

* **Industrial Standard:** ISO 13374 Condition Monitoring and Diagnostics
* **Input Dimensions:** 6 Physical Sensor Channels (Temperature, Speed, Torque, Wear, Vibration, Acoustic dB)
* **Engineered Features:** 9 Tabular Physics Features + 92 Time-Series Window Features
* **Model Classifiers Benchmark:** Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting
* **Probability Calibration:** Isotonic Regression (`CalibratedClassifierCV`)
* **Decision Optimization:** Asymmetric Loss Matrix ($C_{FN} = \$36,000$, $C_{FP} = \$1,200$)
* **Performance Benchmark:**
  * Real UCI AI4I 2020: ROC-AUC **0.978**, PR-AUC **0.871**, F1 **0.840**
  * Unseen Virtual Fleet: ROC-AUC **0.9998**, PR-AUC **0.995**, F1 **0.958**
  * Advance Warning Notice: **54 minutes average (100% of breakdowns flagged)**
* **Plant Financial Impact:** **$1,192,000 / year Net Savings** (55% downtime reduction, 2-day capital payback)
* **Serving & Edge Inference:** Flask REST Microservice (< 12ms latency on Raspberry Pi 4 CPU)

---

## 11.3 Project Navigation Index

1. [`docs/01_project_explanation.md`](01_project_explanation.md) — Simple & Technical explanations, failure modes
2. [`docs/02_tech_stack.md`](02_tech_stack.md) — Architecture decisions & why tree ensembles win
3. [`docs/03_architecture.md`](03_architecture.md) — 7-layer architecture diagram & leak-free splitting
4. [`docs/04_implementation_plan.md`](04_implementation_plan.md) — 6 implementation milestones
5. [`docs/05_installation.md`](05_installation.md) — Windows, macOS, Linux, and Docker setup
6. [`docs/06_virtual_simulation.md`](06_virtual_simulation.md) — Physics degradation equations and dynamics
7. [`docs/07_run_and_outputs.md`](07_run_and_outputs.md) — Description of all 17 figures and output CSVs
8. [`docs/08_github_strategy.md`](08_github_strategy.md) — Live GitHub Pages demo configuration
9. [`docs/09_7day_proof_plan.md`](09_7day_proof_plan.md) — 7-day study & interview preparation schedule
10. [`docs/10_interview_qa.md`](10_interview_qa.md) — 90-second pitch and 10 technical Q&As
11. [`docs/TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — Common runtime exceptions and verified solutions
12. [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — Detailed technical module map
13. [`docs/MODEL_CARD.md`](MODEL_CARD.md) — Model metrics, fairness, and operating boundaries
14. [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) — Edge gateway and production cloud deployment
