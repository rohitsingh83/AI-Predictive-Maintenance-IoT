# 🧯 Troubleshooting & FAQ Guide

Everything you might encounter when running, training, simulating, or deploying the **AI-Powered Predictive Maintenance System for IoT Devices** — with exact, verified solutions.

---

## 📋 Table of Contents
1. [Environment & Python Setup](#environment--python-setup)
2. [Data Generation & Dataset Loading](#data-generation--dataset-loading)
3. [Model Training & Calibration](#model-training--calibration)
4. [Real-Time Simulator & Alert Engine](#real-time-simulator--alert-engine)
5. [Flask REST API & Dashboard](#flask-rest-api--dashboard)
6. [Hardware & Edge Node (Raspberry Pi / ESP32)](#hardware--edge-node-raspberry-pi--esp32)
7. [Testing & CI Pipeline](#testing--ci-pipeline)

---

## 1. Environment & Python Setup

### `python: command not found` / `python3` works but `python` doesn't
* **Linux / macOS:** Use `python3` or alias it in your shell profile (`alias python=python3`).
* **Windows:** Ensure Python is added to your system `PATH`. Check "Add Python to PATH" during installation from [python.org](https://www.python.org/).

### `Activate.ps1 cannot be loaded because running scripts is disabled` (Windows PowerShell)
PowerShell blocks script execution by default. Run the following command in an elevated or current terminal session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### `pip install` installs into the global Python instead of `.venv`
Always activate your virtual environment first, or use the venv's python directly:
```bash
# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Linux / macOS
.venv/bin/python -m pip install -r requirements.txt
```
To verify which Python environment is active:
```bash
python -c "import sys; print(sys.executable)"
```
It should print a path containing `.venv`.

### `ImportError: DLL load failed while importing ... An Application Control policy has blocked this file` (Windows)
Windows Smart App Control or Defender Application Control can block freshly extracted `.pyd` or `.dll` binary extensions with a "Mark-of-the-Web" zone identifier. Unblock them with:
```powershell
Get-ChildItem -Path (python -c "import sys; print(sys.prefix)") -Recurse -Include '*.pyd','*.dll' | Unblock-File
```

### `ModuleNotFoundError: No module named 'src'`
Run commands from the project root directory (`AI-Predictive-Maintenance-IoT`).
If executing from an external script or notebook, ensure the root path is included:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
```

---

## 2. Data Generation & Dataset Loading

### `FileNotFoundError: data/raw/ai4i2020.csv not found`
The UCI AI4I 2020 Predictive Maintenance dataset is downloaded automatically if missing. If automatic download fails due to proxy or firewall restrictions:
1. Download `ai4i2020.csv` directly from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
2. Place the file at `data/raw/ai4i2020.csv`.
3. Alternatively, run:
```bash
python main.py simulate
```
This generates the physics-based synthetic virtual fleet stream without external downloads.

### `How do I regenerate the virtual sensor fleet stream?`
To regenerate 27,000 samples across 30 industrial machines with fresh degradation trajectories:
```bash
python main.py simulate
```
This writes fresh telemetry to `data/raw/iot_sensor_stream.csv`.

---

## 3. Model Training & Calibration

### `Missing models/stream_failure_model.pkl. Run: python main.py train`
Before running the dashboard, live inference, or API, train the model bundles:
```bash
python main.py train
```
For a fast training run (useful for quick smoke tests without extensive grid tuning):
```bash
python main.py train --fast
```

### What is the difference between AI4I Tabular Model and Time-Series Stream Model?
* **AI4I Model (`predictive_maintenance_model.pkl`):** Trained on static point-in-time sensor readings (air temp, process temp, speed, torque, tool wear) with 9 engineered physics features.
* **Stream Model (`stream_failure_model.pkl`):** Trained on dynamic time-series sensor windows across rotating machinery (rolling mean, rolling std, trends, baseline z-scores over 5- and 15-cycle windows). Designed for real-time edge streaming.

### Why do we use Isotonic Probability Calibration?
Standard tree ensembles (Random Forest, HistGradientBoosting) output uncalibrated scores that tend to bunch up near 0 or 1. Isotonic regression maps these raw scores to true empirical probabilities, ensuring that an alarm threshold of `p = 0.52` directly corresponds to a 52% actual probability of breakdown.

---

## 4. Real-Time Simulator & Alert Engine

### How to test the "Live Factory" without physical hardware?
Run the built-in streaming simulator in a terminal:
```bash
python realtime_simulator.py
```
This simulates live machines pushing 10-minute sensor telemetry cycles into the REST API at `http://localhost:5000/stream/reading`, triggering automatic work orders when pre-failure signatures are detected.

### Preventing Alert Fatigue: How does the Alert Engine avoid duplicate alarms?
The `AlertEngine` (`src/alert_system.py`) implements stateful de-duplication:
1. **Cooldown Window:** Once an alert is raised for a machine, subsequent warnings in the same severity band are suppressed for a configurable cycle window.
2. **Escalation Override:** If a condition escalates from `HIGH` to `CRITICAL`, the alert immediately pierces through the cooldown and raises an emergency work order.
3. **Recovery Reset:** If the machine returns to `NORMAL` health for 3 consecutive cycles, the alarm latch automatically resets.

---

## 5. Flask REST API & Dashboard

### `Address already in use: Port 5000`
Another process (or Apple AirPlay on macOS) is using port 5000.
* **Find and kill the process (Windows):**
  ```powershell
  netstat -ano | findstr :5000
  taskkill /PID <PID> /F
  ```
* **Or run Flask on a different port:**
  ```bash
  python api/app.py --port 5050
  ```

### How to access the self-contained operator console?
Run:
```bash
python main.py dashboard
```
Then double-click `outputs/dashboard.html` or open it directly in any modern browser. All plots and metric cards are embedded as base64 images, allowing the file to function 100% offline.

---

## 6. Hardware & Edge Node (Raspberry Pi / ESP32)

### `ModuleNotFoundError: No module named 'Adafruit_DHT'`
The real hardware edge client (`hardware/raspberry_pi_sensor_node.py`) is intended for Raspberry Pi devices with GPIO pins.
On a standard desktop/laptop, physical hardware is completely optional because `src/simulate_iot_sensors.py` generates high-fidelity simulated telemetry.
If configuring a physical Raspberry Pi:
```bash
sudo apt-get update
sudo apt-get install python3-pip python3-dev
pip3 install Adafruit_DHT requests
```

### Sensor Pinout & Voltage Levels
* **DHT22 (Temperature/Humidity):** 3.3V or 5V VCC, GPIO 4 (Pin 7). Requires a 10kΩ pull-up resistor between VCC and Data.
* **SW-420 (Vibration):** 3.3V VCC, GPIO 17 (Pin 11). Digital output triggers on mechanical shock.
* **ACS712 (Current Sensor):** Requires an analog-to-digital converter (ADC) such as ADS1115 (I2C: SDA Pin 3, SCL Pin 5) to interface with Raspberry Pi 3.3V GPIO safely.
* **Emergency Relay Interlock:** 5V VCC, GPIO 27 (Pin 13). Active-low trigger disconnects spindle power on `CRITICAL` alert.

---

## 7. Testing & CI Pipeline

### How to run the automated test suite?
Run:
```bash
.venv\Scripts\pytest.exe -v
```
All 10 tests verify:
1. Configuration paths and feature sets.
2. Physics-based simulator boundaries and degradation trajectory dynamics.
3. Data cleaning and deduplication logic.
4. Leak-free feature engineering contract.
5. Threshold optimization and cost matrix calculations.
6. Alert engine severity escalation and de-duplication.
7. Model artifact bundle schema and serializability.
8. Flask API `/health` endpoint response.
9. Flask API `/predict` tabular scoring endpoint.
10. Flask API stream window scoring contract.

---

*For further questions, open an issue at [https://github.com/rohitsingh83/AI-Predictive-Maintenance-IoT/issues](https://github.com/rohitsingh83/AI-Predictive-Maintenance-IoT/issues).*
