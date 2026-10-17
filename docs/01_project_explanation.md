# 1. Project Explanation — AI Predictive Maintenance for IoT Devices

---

## 1.1 What Is Predictive Maintenance? (Simple Language)

Imagine a factory that produces car engines using 30 heavy-duty industrial milling machines. Each machine runs 16 hours a day.

In traditional manufacturing, maintenance happens in one of two ways:
1. **Reactive Maintenance (Fix It When It Breaks):** You wait until the machine emits smoke, screeches, and catastrophically breaks. This causes **unplanned downtime** — workers stand idle, production lines freeze, and emergency repairs cost **$36,000+ per incident**.
2. **Preventive Maintenance (Calendar-Based):** You service the machine every 30 days regardless of whether it actually needs maintenance. This wastes money on replacement parts that still had weeks of remaining life.

**AI-Powered Predictive Maintenance (PdM)** is the intelligent third way:
Sensors attached to the machinery constantly measure temperature, vibration, rotational speed, torque, and acoustic noise. An AI model analyzes these sensor streams in real time, learns the subtle physical pre-failure signatures, and sounds an alarm **~1 hour before the machine would fail**.

This gives maintenance crews enough advance notice to service the tool during an scheduled break, saving **$1,190,000+ per year** across a 30-machine plant.

---

## 1.2 What Is It? (Technical Language)

Predictive Maintenance (PdM) is a cyber-physical machine learning framework compliant with **ISO 13374 (Condition Monitoring and Diagnostics of Machines)**.

The system processes multi-channel multivariate IoT telemetry (thermal, kinematic, electrical, acoustic) across industrial assets. The task is formulated as a binary classification problem under extreme class imbalance (3.4% failure rate):

$$\hat{y} = f(X_t, X_{t-W:t}, \Phi(X_t)) \in \{0, 1\}$$

where:
* $X_t$ is the instantaneous sensor vector.
* $X_{t-W:t}$ represents rolling-window summary statistics (mean, variance, skew, trend) over 5- and 15-cycle historical windows.
* $\Phi(X_t)$ represents domain-specific physics transformations ($P_{mech} = \tau \cdot \omega$, thermal dissipation index $\Delta T$, tool wear interaction).
* Predictions are calibrated via **non-parametric Isotonic Regression** to produce honest posterior probabilities $P(\text{Failure} \mid X)$.
* Decisions are thresholded against an asymmetric **business cost loss matrix** ($C_{FN} = \$36,000$ vs $C_{FP} = \$1,200$).

---

## 1.3 Key Failure Modes Detected

The system identifies five distinct industrial degradation mechanisms:

| Failure Mode | Physical Root Cause | Key Sensor Indicators |
|---|---|---|
| **Tool Wear Failure (TWF)** | Progressive cutting insert flank wear exceeding critical carbide friction limit | Accumulated tool wear > 240 min, high cutting torque, harmonic chatter |
| **Heat Dissipation Failure (HDF)** | Coolant restriction, radiator clogging, or air convection deficit | Process-to-air differential $\Delta T < 8.6\text{K}$ at low rotational speed |
| **Power Failure (PWF)** | Spindle electrical drive overload or mechanical stall | Mechanical power $P_{mech} = \tau \cdot \omega < 3,500\text{W}$ or $> 9,000\text{W}$ |
| **Overstrain Failure (OSF)** | Catastrophic structural overload under high cutting torque and tool wear | Interaction product $\text{Tool Wear} \times \text{Torque} > 11,000\text{ min}\cdot\text{Nm}$ |
| **Bearing Wear / Imbalance** | Incipient spalling or race defects in rotating element bearings | RMS vibration velocity $> 4.5\text{ mm/s}$, acoustic emission spike $> 75\text{ dB}$ |

---

## 1.4 End-to-End System Workflow

```
[IoT Sensor Nodes] (Temp, Vib, RPM, Torque, Current)
         │
         ▼
[Edge Gateway] (Validation, Buffering, Relay Cutoff)
         │
         ▼
[Data Cleaning & Feature Pipeline] (Physics features + 92 rolling window features)
         │
         ▼
[Calibrated Classifier] (HistGradientBoosting / Random Forest + Isotonic Regression)
         │
         ▼
[Stateful Alert Engine] (De-duplication, Cooldown, Severity Latches)
         │
         ▼
[Automated Work Order Dispatch] (SAP PM / Maximo JSON + Dashboard Alert)
```
