# 9. 7-Day Mastery & Proof of Work Plan

---

## 9.1 Day-by-Day Roadmap for Candidates & Students

| Day | Focus Area | Hands-on Objective | Key Deliverables |
|---|---|---|---|
| **Day 1** | **Problem & Data Ingestion** | Understand industrial failure costs, explore AI4I dataset in `notebooks/01_data_exploration.ipynb` | `data/raw/ai4i2020.csv` inspected; class imbalance understood |
| **Day 2** | **Physics Simulation** | Run `src/simulate_iot_sensors.py`, inspect degradation dynamics and sensor noise | 27,000 multi-channel samples generated in `data/raw/` |
| **Day 3** | **Feature Engineering** | Review physics transformations and 92 rolling window features in `notebooks/02_feature_engineering.ipynb` | No-leakage test verified via `pytest tests/test_pipeline.py` |
| **Day 4** | **Model Benchmark & Training** | Train 4 models, compare ROC/PR curves in `notebooks/03_model_training_evaluation.ipynb` | `models/stream_failure_model.pkl` generated with metrics |
| **Day 5** | **Alert Engine & Work Orders** | Inspect `src/alert_system.py`, test severity bands, cooldown, and de-duplication | `outputs/live_alerts.csv` populated with work orders |
| **Day 6** | **Dashboard & Web Platform** | Open `outputs/dashboard.html` and `docs/index.html`, test interactive failure presets | Screenshots captured via `scripts/capture_screenshots.py` |
| **Day 7** | **Interview Pitch & ROI Defense** | Rehearse the 90-second elevator pitch and defend the $1.19M financial business case | Ready for technical interviews and project defense |
