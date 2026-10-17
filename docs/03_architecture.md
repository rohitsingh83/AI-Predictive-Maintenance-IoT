# 3. System Architecture & End-to-End Pipeline

---

## 3.1 7-Layer Architecture Overview

```mermaid
flowchart TD
    subgraph L1["Layer 1: Sensor Ingestion & Telemetry"]
        S1["Temperature (DHT22 / PT100)"]
        S2["Vibration (SW-420 / Piezo)"]
        S3["Spindle Current (ACS712 / Hall)"]
        S4["Rotational Speed & Torque (Rotary Encoder)"]
    end

    subgraph L2["Layer 2: Edge Gateway"]
        GW["Raspberry Pi 4 / ESP32 Gateway"]
        BUF["Circular Rolling Buffer (15 cycles)"]
        REL["Emergency Relay Cutoff Interlock"]
    end

    subgraph L3["Layer 3: Data Pipeline & Hygiene"]
        CLN["Range Clamping & Drop-out Interpolation"]
        SPL["Machine-Wise Split (GroupKFold - Leak-free)"]
    end

    subgraph L4["Layer 4: Feature Engineering"]
        PHY["Physics Transformations (P_mech, Delta-T)"]
        WIN["92 Rolling Window Stats (mean, std, min, max, trend)"]
    end

    subgraph L5["Layer 5: Machine Learning Core"]
        HGB["HistGradientBoosting / Random Forest"]
        ISO["Isotonic Probability Calibration"]
        OPT["Cost-Optimal Threshold Tuner (tau = 0.52)"]
    end

    subgraph L6["Layer 6: Stateful Alert Engine"]
        LAT["Severity Latches (LOW -> CRITICAL)"]
        DED["De-duplication & Cooldown Logic"]
        WO["Automated SAP / Maximo Work Order Generator"]
    end

    subgraph L7["Layer 7: Operator Dashboard & API"]
        API["Flask REST API (/predict, /stream/reading)"]
        DSH["Interactive HTML5 Digital Twin Console"]
        REP["Business Impact ROI Report ($1.19M/yr)"]
    end

    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> L5
    L5 --> L6
    L6 --> L7
    L6 -.->|Emergency Interlock Actuation| REL
```

---

## 3.2 Leak-Free Machine-Wise Partitioning

A critical flaw in many time-series projects is randomly shuffling observations across the train and test split. When consecutive cycles from the same machine appear in both sets, the model simply memorizes the individual machine's baseline vibration signature.

**Our Architecture Enforces:**
* Partitioning strictly **by Machine ID** (`GroupKFold`).
* The model is tested exclusively on **machines it has never seen during training**.
* Result on unseen machines: **PR-AUC 0.995**, **ROC-AUC 0.9998**, and **100% of catastrophic breakdowns flagged in advance** with ~54 minutes of notice.
