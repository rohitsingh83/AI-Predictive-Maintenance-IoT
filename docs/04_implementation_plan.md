# 4. Implementation Plan & Milestones

---

## 4.1 Milestone Breakdown

The project was constructed across 6 systematic implementation phases:

### Phase 1: Physical Data Modeling & Fleet Simulation
* Implemented the degradation physics simulator in [`src/simulate_iot_sensors.py`](file:///c:/Users/Rohit%20Singh/OneDrive/Desktop/IITD%20PROJECTS/AI%20Projects/AI-Predictive-Maintenance-IoT/src/simulate_iot_sensors.py).
* Simulated 30 virtual machines generating 27,000 multi-channel samples with 118 authentic breakdown events.
* Integrated the real UCI AI4I 2020 dataset (10,000 industrial cycles) with automatic fallback download.

### Phase 2: Leak-Free Preprocessing & Feature Engineering
* Built physical domain features: mechanical power, thermal differential, tool wear interaction.
* Implemented time-series window aggregations: rolling mean, standard deviation, min, max, range, and trend over 5 and 15 cycle horizons.
* Enforced machine-wise GroupKFold cross-validation partitioning.

### Phase 3: Model Benchmark, Calibration & Threshold Optimization
* Benchmarked 4 classifiers: Logistic Regression, Random Forest, Extra Trees, and HistGradientBoosting.
* Fitted non-parametric Isotonic Regression to map raw scores to empirical probabilities.
* Tuned the decision threshold against the asymmetric loss matrix ($C_{FN} = \$36,000$, $C_{FP} = \$1,200$), arriving at the cost-optimal cutoff ($\tau = 0.52$).

### Phase 4: Stateful Alert Engine & Work Order Automation
* Implemented severity categorization: `NORMAL`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`.
* Added cooldown latches to suppress redundant alarms within the same severity level.
* Built automated SAP PM / IBM Maximo work order payload generator with technician repair instructions.

### Phase 5: Edge Gateway Daemon & REST API Microservice
* Developed the Flask REST API exposing `/predict`, `/stream/reading`, `/health`, and `/alerts`.
* Created the Raspberry Pi edge daemon (`hardware/raspberry_pi_sensor_node.py`) with GPIO pinout and emergency relay cutoff logic.

### Phase 6: Live Web Application, Documentation & Testing
* Created the interactive HTML5 digital twin platform (`docs/index.html` and `outputs/dashboard.html`).
* Developed the automated Selenium screenshot generator (`scripts/capture_screenshots.py`).
* Wrote 10 comprehensive pytest tests verifying physical ranges, leakage guards, metrics, and API routes.
