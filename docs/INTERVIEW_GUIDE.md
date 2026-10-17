# Interview & Placement Guide

Everything you need to defend this project confidently, plus 40 questions an
interviewer may ask (with model answers).

## 1. The 90-second pitch (memorise this)

> "I built an end-to-end predictive maintenance system for IoT-instrumented machines.
> It takes six sensor channels - temperature, vibration, current, pressure, rpm and
> acoustic emission - and predicts a breakdown *before* it happens. I trained two
> models: one on the real AI4I 2020 industrial dataset from UCI, where I reached a
> 0.978 ROC-AUC and 0.81 recall with 0.87 precision after engineering physics features
> like mechanical power and heat-dissipation index, and one on a virtual 30-machine
> IoT fleet where rolling-window features gave a 0.958 F1 on machines the model had
> **never seen**. Because failures are rare, I tuned the alarm threshold by expected
> business cost instead of accuracy - a missed failure costs about 16 times a false
> alarm - which the alert engine converts into work orders with severity bands. It's
> deployed as a Flask API with a self-contained HTML dashboard, and there's an optional
> Raspberry Pi sensor node that closes the loop by cutting motor power on a critical
> alert. The whole thing runs from a single `python main.py all` command and the CI
> runs the tests on every push."

## 2. Talking points that separate you from other students

| Weak project says | This project says |
|---|---|
| "I got 99% accuracy" | "Accuracy is meaningless at 3% failure rate - I report PR-AUC, recall, MCC and Brier" |
| "I split 80/20 randomly" | "Random splitting leaks machine identity; I split **by machine** to answer 'will it work on a new machine?'" |
| "I used 0.5 as the threshold" | "The threshold is a business decision - I minimise expected cost: 8 h downtime vs a 0.5 h inspection" |
| "The model outputs a probability" | "Probabilities are **isotonic-calibrated**, so the risk bands mean what they say" |
| "I plotted a confusion matrix" | "Every cell is labelled with its business consequence, and translated to dollars" |
| "I don't have hardware" | "I built a documented virtual fleet with a hidden health index, and a Raspberry Pi node that runs the same code path when hardware exists" |

## 3. 40 likely questions (short answers)

**Data & problem framing**
1. *Why is failure prediction hard?* Class imbalance (2-3%), overlapping sensor
   distributions, few labelled failures, and non-stationary machines.
2. *Where do labels come from in reality?* Maintenance work orders / CMMS - the AI
   learns what happened before each logged failure.
3. *Why not anomaly detection instead?* Anomaly detection finds "unusual" - not all
   unusual behaviour is a failure. Supervised classification answers the actual question.
4. *What is the prediction horizon and why?* 6 cycles = 1 hour here - long enough to
   schedule a repair, short enough to stay accurate. Tune it to the repair lead time.
5. *How would you get more failure data?* Physics simulation, augmentation of rare
   modes, or transfer learning from a similar machine.

**Features**
6. *Why engineer features when trees can learn?* Trees cannot compute torque x rpm;
   giving them physical quantities measurably improves performance and interpretability.
7. *What does the heat-dissipation index capture?* (process - air temp) normalised by
   power - poor cooling efficiency, the HDF signature.
8. *Why rolling windows?* Degradation is about change over time, not a single sample.
9. *Why baseline z-scores?* Each machine has its own "normal"; a fleet-wide threshold
   flags healthy-but-noisy machines.
10. *How do you avoid leakage?* Explicit feature whitelist; ground-truth columns and
    baselines excluded; machine-wise split; scaling fitted inside CV folds.

**Model & evaluation**
11. *Why compare 4 models?* To have evidence, and to catch a weak result early.
    Logistic regression is the honest baseline that trees must beat.
12. *Why did RandomForest win on AI4I?* It captures the non-linear overstrain
    interaction (low rpm + high torque) and handles mixed scales without scaling.
13. *What is PR-AUC and why care?* Area under precision-recall curve; the honest metric
    when positives are rare, and directly tunable via the threshold.
14. *ROC-AUC 0.978 vs PR-AUC 0.871 - why the gap?* ROC uses the huge true-negative pool;
    PR only looks at the rare positives, so it is far more demanding.
15. *What's MCC?* Correlation between predictions and truth across all four cells -
    robust under imbalance, unlike accuracy.
16. *Why calibrate?* To make the risk bands meaningful and enable expected-cost maths.
17. *What is Brier score?* Mean squared error of probabilities - measures calibration.
18. *How do you handle class imbalance?* Class weights (always), optional SMOTE,
    cost-based threshold tuning (the big one).
19. *Why not LSTM?* With 27k samples of engineered degradation, trees matched/exceeded
    an LSTM at a fraction of the cost; LSTM+GPU is documented as the advanced extension.
20. *Overfitting check?* CV vs test gap, PR-AUC both ways, and identical feature
    contract at inference.

**Business**
21. *How much does the system save?* Expected scenario: ~$1.19M/yr on a 30-machine line
    (55% cost reduction); conservative scenario still saves ~$0.68M - inside the
    15-30% downtime-reduction range published by industry studies.
22. *Why is payback so fast?* The retrofit is only per-machine sensors + edge ($250);
    the savings come from avoided downtime, not from new machinery.
23. *What if the plant can't act on alarms?* Then only the conversion rate drops - that
    is exactly what the conservative scenario models.
24. *Biggest cost risk?* False alarms causing unnecessary line stops, which is why the
    cost model charges 0.5 h of downtime per false alarm, not just inspection time.

**Engineering**
25. *How do you serve the model?* Flask API with three endpoints (snapshot, window,
    single reading) plus a rolling per-machine buffer.
26. *How do you keep train/serve consistent?* One feature engineering module used by
    both training and inference; the bundle stores the exact feature list.
27. *What happens when a sensor fails?* Median imputation at train time; range
    validation and drop-outs handled at the edge; explicit "no telemetry" row removal.
28. *How would you monitor in production?* Input-distribution drift, prediction rate
    per severity, alert-to-work-order conversion, and realised vs predicted failures.
29. *Deployment target?* Laptop/Docker for the demo; Raspberry Pi edge node (code
    included) for the physical rig; cloud broker + container for scale.
30. *How do you version models?* The saved bundle records model name, threshold, features,
    seed and `trained_at`; a model card documents each release.

**Stretch / judgement**
31. *What would you do with 10x more data?* Sequence models (LSTM/TCN), RUL regression,
    per-component models (bearing vs motor), and survival analysis.
32. *Where does this fail?* Sudden catastrophic failures (RNF-style) have no precursor -
    no model can predict them; you need protective interlocks instead.
33. *How would you prioritise repairs across a fleet?* Rank by probability x cost of
    downtime x criticality of the asset - this repo's fleet health map is the start.
34. *Explain this to a plant manager in one sentence.* "Instead of waiting for a machine
    to break, the system reads its vitals and gives your crew a work order about an hour
    before it would have failed."
35. *What is the weakest part of the project?* The virtual fleet's physics is simulated,
    and the cost constants are illustrative - both are documented and easy to replace.
36. *Which metric would you put on a wall dashboard?* % of breakdowns predicted in
    advance (100% here) and average lead time (0.9 h) - the two numbers the crew feels.
37. *How do you handle multiple failure modes?* Currently binary; the dataset labels
    (TWF/HDF/PWF/OSF) enable one-vs-rest multi-label classification as an extension.
38. *What is PM vs PdM vs prescriptive maintenance?* Scheduled vs predicted vs
    predicted-plus-optimal-action.
39. *Any security concerns?* Unsigned MQTT traffic and unauthenticated API in the demo;
    production needs TLS, device certificates, API keys and audit logs.
40. *Why should we hire you based on this?* Because I can take an ambiguous industrial
    problem, make honest modelling decisions under data and hardware constraints, turn
    results into money and actions, and ship it as a tested, documented, runnable system.

## 4. 5-minute live demo script

1. `python main.py all` - say: "one command reproduces everything, 4 minutes"
2. Open `outputs/dashboard.html` - point at the KPI row, then press **Play replay** and
   let the gauge climb while the alarm log fills; say: "this is a machine the model never
   saw, and the work order appears before the breakdown".
3. `python api/app.py` in one terminal,
   `python realtime_simulator.py --machines 3 --interval 1 --accentuate 8` in another -
   say: "this is what the shop floor sees; the edge posts readings and the API returns
   severity + a work order".
4. Show `reports/business_impact.md` - "and here is what it is worth per year".
5. Finish with `images/10_early_warning_timeline.png` - "0.9 hours of advance warning,
   measured on unseen machines."

## 5. Resume bullets (copy-paste)

* Built an end-to-end **AI predictive-maintenance system** for IoT-instrumented machinery
  (Python, scikit-learn, Flask) achieving **0.978 ROC-AUC / 0.87 precision** on the
  AI4I 2020 industrial dataset and **0.958 F1** on a 30-machine virtual fleet with
  **0.9 h average early-warning lead time** on unseen machines.
* Engineered 9 physics-based and 80+ rolling-window features; eliminated data leakage via
  machine-wise splitting; replaced default 0.5 threshold with **cost-optimised decision
  thresholds** derived from downtime economics ($36k per missed failure vs $2.3k false alarm).
* Delivered a **Flask REST API**, a self-contained **HTML operations dashboard**, an alert
  engine with severity banding/de-duplication, and a **business case showing $1.19M annual
  savings** on a 30-machine line; documented with tests and CI.
