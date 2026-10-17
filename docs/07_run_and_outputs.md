# 7. Pipeline Execution, Figures & Output Artifacts

---

## 7.1 Single Entry Point (`main.py`)

The pipeline can be executed modularly or in a single pass:

```bash
python main.py simulate     # Step 1: Generate 27,000 virtual IoT sensor samples
python main.py train        # Step 2: Train & calibrate both AI models
python main.py evaluate     # Step 3: Compute PR/ROC-AUC, confusion matrices, model cards
python main.py visualize    # Step 4: Generate all 17 publication-quality figures
python main.py report       # Step 5: Generate finance-ready business impact report
python main.py dashboard    # Step 6: Build self-contained outputs/dashboard.html
python main.py all          # Execute entire pipeline end-to-end (~2 minutes)
```

---

## 7.2 Directory of Generated Figures ([`images/`](file:///c:/Users/Rohit%20Singh/OneDrive/Desktop/IITD%20PROJECTS/AI%20Projects/AI-Predictive-Maintenance-IoT/images))

| Figure | Description |
|---|---|
| `01_dataset_overview.png` | AI4I 2020 distributions, failure mode counts, and sensor ranges |
| `02_sensor_correlation.png` | Spearman correlation matrix highlighting physical interactions |
| `03_model_comparison.png` | Bar chart comparing ROC-AUC, PR-AUC, and F1 across 4 classifiers |
| `04_confusion_matrix.png` | Cost-weighted confusion matrix @ threshold 0.26 with business labels |
| `05_roc_pr_curves.png` | Receiver Operating Characteristic and Precision-Recall curves |
| `06_feature_importance.png` | Top predictive drivers (Mechanical Power, Wear×Torque, $\Delta T$) |
| `07_threshold_optimisation.png` | Total plant operating cost as a function of decision threshold |
| `08_probability_distribution.png` | Uncalibrated vs Isotonically-calibrated probability distributions |
| `09_degradation_trajectory.png` | Run-to-failure degradation progression across machine cycles |
| `10_early_warning_timeline.png` | Alert trigger cycle vs failure cycle demonstrating 54-min notice |
| `11_fleet_health_map.png` | Heatmap of 30 machines showing operational health across time |
| `12_eda_stream_overview.png` | Summary statistics of multi-channel virtual sensor telemetry |
| `13_confusion_matrix_stream.png` | Test-set confusion matrix on unseen machines |
| `14_model_comparison_stream.png` | Model benchmark on streaming time-series features |
| `15_roc_pr_stream.png` | ROC and PR curves on held-out machines (PR-AUC 0.995) |
| `16_feature_importance_stream.png` | Top drivers for time-series model (rolling std, trend, z-score) |
| `17_business_impact.png` | Net savings ($1.19M), downtime hours saved, and payback period |

---

## 7.3 Operational Output Artifacts ([`outputs/`](file:///c:/Users/Rohit%20Singh/OneDrive/Desktop/IITD%20PROJECTS/AI%20Projects/AI-Predictive-Maintenance-IoT/outputs))

* **`dashboard.html`:** Standalone HTML operations console with base64 embedded figures and JavaScript live replay.
* **`live_alerts.csv`:** Full log of every work order raised by the alert engine during simulation.
* **`alert_lead_time_report.csv`:** Per-breakdown lead time accounting (mean: 5.3 cycles = 0.9 hours notice).
* **`test_predictions.csv`:** Scored probabilities and ground truth for test-set machines.
* **`business_impact.json`:** Structured financial metrics used for executive dashboards.
