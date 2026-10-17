# 2. Tech Stack & Architectural Decisions

---

## 2.1 Technology Stack Architecture

| Layer | Technology | Rationale |
|---|---|---|
| **Core Language** | Python 3.9 – 3.12 | Ubiquitous industrial standard with comprehensive scientific and embedded ecosystem |
| **Data Processing** | NumPy, Pandas | High-performance vectorized operations on multi-channel time-series data |
| **Machine Learning** | scikit-learn 1.3+ | Fast, deterministic CPU training; HistGradientBoosting, Random Forest, Extra Trees |
| **Probability Calibration** | CalibratedClassifierCV | Isotonic regression maps tree output rankings to empirical failure probabilities |
| **Feature Engineering** | Custom Physics Engine | Mechanical power ($P = \tau \cdot \omega$), heat dissipation $\Delta T$, rolling window aggregations |
| **Serving & API** | Flask 3.0+ | Lightweight, low-overhead REST microservice suitable for edge IPCs and gateways |
| **Data Serialization** | Joblib | High-efficiency serialisation of calibrated pipeline bundles |
| **Edge Hardware** | Raspberry Pi 4 / ESP32 | Standard Industrial IoT gateway hardware with GPIO sensor interfaces |
| **Frontend Platform** | Vanilla HTML5 / Canvas / CSS3 | Zero-dependency, offline-capable digital twin console and GitHub Pages demo |
| **Automated Testing** | Pytest | 10 unit and integration tests verifying physics invariants and API contracts |

---

## 2.2 Why Tree Ensembles Over Deep Learning (LSTM / Transformers)?

For industrial IoT sensor streams with tabular physics and rolling window features, **HistGradientBoosting and Random Forest** significantly outperform deep recurrent neural networks (LSTMs):

1. **Inference Latency on Low-Cost Hardware:**
   * Tree ensembles execute in `< 12ms` on an unaccelerated Raspberry Pi CPU.
   * Deep LSTMs or Transformers require $> 150\text{ms}$ or dedicated GPU/NPU hardware.
2. **Data Efficiency & Class Imbalance:**
   * Industrial failures are rare (3.4% prevalence). Gradient boosting with balanced class weights handles severe skew far more stably than neural networks which tend to overfit or collapse without synthetic oversampling.
3. **Interpretability & Explainability:**
   * Tree ensembles provide exact feature importances, allowing plant engineers to pinpoint whether an alarm was driven by torque, thermal deficit, or vibration spike.

---

## 2.3 Physics-Informed Feature Engineering vs Raw Feature Feeding

Feeding raw sensor data directly into a model forces the algorithm to independently rediscover physical laws. By explicitly engineering domain-specific physics features:
* **Mechanical Power ($W$):** $P_{mech} = \tau \times \frac{2\pi \cdot \text{RPM}}{60}$
* **Thermal Dissipation Index:** $\Delta T = T_{process} - T_{air}$
* **Wear-Torque Interaction:** $\text{Tool Wear} \times \tau$
* **Kinetic Vibration Energy:** Rolling standard deviation $\sigma_{vib}$ over 5 and 15 cycles

The model achieves **ROC-AUC 0.978** and **PR-AUC 0.871** on real industrial benchmark data.
