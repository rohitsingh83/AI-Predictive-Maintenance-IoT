# 10. Technical Interview Q&A & Project Defense

---

## 10.1 The 90-Second Executive Pitch

> *"In manufacturing, unplanned machine downtime costs $36,000+ per hour. Traditional maintenance is either reactive — waiting until equipment catastrophically breaks — or preventive on a calendar schedule, which replaces parts that still have weeks of usable life.*
> 
> *I built an end-to-end AI Predictive Maintenance system for industrial IoT machinery. It monitors 6 physical sensor streams — temperature, vibration, acoustic emission, pressure, torque, and rotational speed. By engineering physical domain features like mechanical power ($P = \tau \cdot \omega$) and thermal dissipation differentials, paired with rolling time-series statistics over 5- and 15-cycle windows, the system flags impending failures ~54 minutes in advance.*
> 
> *I benchmarked four models using leak-free GroupKFold cross-validation by machine ID. The champion HistGradientBoosting classifier achieves 0.978 ROC-AUC on real industrial data and 0.995 PR-AUC on unseen virtual fleet machines. Using isotonic calibration and an asymmetric cost matrix, the system generates automated SAP work orders and delivers $1.19M in annual net savings for a 30-machine plant with a 2-day capital payback."*

---

## 10.2 Top 10 High-Frequency Technical Questions

### Q1: Why use PR-AUC instead of standard accuracy?
**Answer:** In predictive maintenance, failures represent only 3.4% of all cycles. A naive model that always predicts "healthy" would achieve 96.6% accuracy while missing 100% of catastrophic failures. Precision-Recall AUC (PR-AUC) evaluates model performance strictly on the positive (failure) class, penalizing false positives and false negatives without being inflated by the large majority class of healthy cycles.

### Q2: How did you ensure your time-series evaluation had zero data leakage?
**Answer:** We partitioned the data strictly **by Machine ID** using `GroupKFold`. If you split cycles randomly, the model memorizes each machine's specific baseline sensor noise and reports artificially high metrics. Our test set consists entirely of machines the model has never encountered during training.

### Q3: Why is probability calibration necessary when you already have decision trees?
**Answer:** Raw tree ensemble predictions are heuristic ranking scores rather than true empirical probabilities; they cluster around extremes. By fitting non-parametric **Isotonic Regression** on an out-of-fold validation set, a predicted probability of 0.60 corresponds directly to a 60% historical probability of failure, making the scores economically actionable.

### Q4: How do you choose the decision threshold ($\tau = 0.52$)?
**Answer:** In manufacturing, the cost of a False Negative ($36,000 in lost production and secondary wreckage) is 30× greater than the cost of a False Positive ($1,200 for a technician inspection). We evaluate total plant cost across a sweep of 100 candidate thresholds from 0.05 to 0.95 and select the minimum-cost cutoff:
$$\text{Cost}(\tau) = C_{FP} \cdot \text{FP}(\tau) + C_{FN} \cdot \text{FN}(\tau)$$

### Q5: How do you prevent alarm fatigue among plant maintenance technicians?
**Answer:** The stateful `AlertEngine` enforces a 6-cycle cooldown window for repeat alerts within the same severity level. However, if conditions worsen from `HIGH` to `CRITICAL`, the system triggers an emergency override and dispatches a priority P1 work order immediately.

### Q6: Can this run on an edge gateway without cloud connectivity?
**Answer:** Yes. The feature extractor and calibrated model bundle are lightweight scikit-learn artifacts that execute in `< 12ms` on a Raspberry Pi 4 CPU, enabling local relay motor cutoffs even during total network severance.

### Q7: What are the main physical failure modes in your model?
**Answer:** Five modes: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Overload (PWF), Overstrain Failure (OSF), and Bearing Imbalance / Resonance.

### Q8: What features had the highest predictive importance?
**Answer:** Mechanical Power ($P = \tau \cdot \omega$), Tool Wear × Torque interaction product, Process-to-Air Temperature differential ($\Delta T$), and 15-cycle rolling standard deviation of vibration velocity ($\sigma_{vib}$).

### Q9: What happens if a sensor drifts over time?
**Answer:** Sensor drift degrades model calibration. In production, we deploy distribution monitoring (Evidently / Kolmogorov-Smirnov statistical tests) to detect covariate drift in sensor baselines and trigger automated sensor recalibration workflows.

### Q10: How is the $1.19M annual net saving calculated?
**Answer:** A 30-machine plant experiencing 2 breakdowns/machine/year suffers 60 failures, causing 480 hours of unplanned downtime totaling $2,160,000 in lost production. With 55% downtime reduction through advance alerts, 34 failures are converted to scheduled $1,200 services, avoiding $1.19M in net losses after accounting for sensor hardware costs.
