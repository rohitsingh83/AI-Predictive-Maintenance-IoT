"""
config.py
=========
Single source of truth for every path, column name, threshold and business
assumption used across the AI-Powered Predictive Maintenance project.

WHY THIS FILE EXISTS
--------------------
Beginners hard-code paths like "C:/Users/xyz/data.csv" everywhere. In industry
you never do that: you keep configuration in ONE place so the code runs on your
laptop, on a Raspberry Pi edge gateway, in Docker and in CI without edits.

Change a value here -> the whole pipeline (simulator, training, API, dashboard)
picks it up automatically.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = BASE_DIR / "models"
OUTPUTS_DIR   = BASE_DIR / "outputs"
IMAGES_DIR    = BASE_DIR / "images"
REPORTS_DIR   = BASE_DIR / "reports"
DOCS_DIR      = BASE_DIR / "docs"

for _p in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR, REPORTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. DATASET FILES
# ---------------------------------------------------------------------------
# (A) REAL industry dataset - AI4I 2020 Predictive Maintenance (UCI ML repo)
#     Source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
#     Licence: CC BY 4.0  |  Attribution: Stephan Matzka, 2020
AI4I_RAW_CSV      = RAW_DIR / "ai4i2020.csv"

# (B) Our own VIRTUAL IoT sensor stream produced by src/simulate_iot_sensors.py
STREAM_RAW_CSV    = RAW_DIR / "iot_sensor_stream.csv"

# Processed / split files
AI4I_TRAIN_CSV    = PROCESSED_DIR / "ai4i_train.csv"
AI4I_TEST_CSV     = PROCESSED_DIR / "ai4i_test.csv"
STREAM_TRAIN_CSV  = PROCESSED_DIR / "stream_train.csv"
STREAM_TEST_CSV   = PROCESSED_DIR / "stream_test.csv"

# ---------------------------------------------------------------------------
# 3. REPRODUCIBILITY
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE   = 0.20            # 80 / 20 split
CV_FOLDS    = 5               # Stratified K-Fold cross validation

# ---------------------------------------------------------------------------
# 4. AI4I 2020 COLUMN MAP  (raw = as downloaded, clean = pythonic names)
# ---------------------------------------------------------------------------
AI4I_COLUMN_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf_tool_wear_failure",
    "HDF": "hdf_heat_dissipation_failure",
    "PWF": "pwf_power_failure",
    "OSF": "osf_overstrain_failure",
    "RNF": "rnf_random_failure",
}

AI4I_SENSOR_COLUMNS = [
    "air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min",
]

# Features ENGINEERED from physics/domain knowledge (see feature_engineering.py)
AI4I_ENGINEERED_COLUMNS = [
    "temp_difference_k",       # process - air  -> poor heat dissipation indicator
    "mechanical_power_w",      # torque x angular speed -> real power load
    "torque_per_rpm",          # overstrain tendency
    "tool_wear_rate",          # wear per unit of mechanical work
    "wear_x_torque",           # interaction: worn tool + high torque = overstrain
    "heat_dissipation_index",  # temp diff normalised by power
    "rpm_deviation",           # rpm away from healthy band (overstrain/heat)
    "is_high_wear",            # binary flag: tool wear > 200 min
    "is_low_rpm_high_torque",  # binary flag: classic overstrain signature
]

AI4I_FEATURES = AI4I_SENSOR_COLUMNS + AI4I_ENGINEERED_COLUMNS + ["type_encoded"]
AI4I_TARGET   = "machine_failure"
AI4I_MODE_COLUMNS = [
    "twf_tool_wear_failure", "hdf_heat_dissipation_failure",
    "pwf_power_failure", "osf_overstrain_failure", "rnf_random_failure",
]
AI4I_TYPE_MAP = {"L": 0, "M": 1, "H": 2}

# ---------------------------------------------------------------------------
# 5. VIRTUAL IoT STREAM COLUMNS
# ---------------------------------------------------------------------------
STREAM_SENSOR_COLUMNS = [
    "temperature_c",       # DHT22-style temperature sensor
    "vibration_mm_s",      # accelerometer / SW-420 style vibration sensor
    "current_a",           # ACS712 style current sensor
    "pressure_bar",        # pressure transducer
    "rpm",                 # tachometer
    "acoustic_db",         # acoustic emission sensor
]

# Rolling-window (time-series) features built in feature_engineering.py
STREAM_WINDOWS = (5, 15)     # short & medium memory of the machine
STREAM_TARGET = "failure_within_horizon"   # 1 if the machine fails within N cycles
STREAM_NOW_TARGET = "failure_now"

# ---------------------------------------------------------------------------
# 6. PREDICTIVE-MAINTENANCE BUSINESS RULES
# ---------------------------------------------------------------------------
FAILURE_HORIZON_CYCLES = 6      # predict failures up to 6 cycles (=1 hour) ahead
SAMPLING_MINUTES       = 10     # one sensor sample every 10 minutes (virtual factory)

# Risk bands: probability -> severity -> recommended action
RISK_BANDS = [
    (0.80, "CRITICAL", "STOP & inspect now - schedule emergency maintenance"),
    (0.55, "HIGH",     "Create work order within 24 h - inspect bearings, lubrication"),
    (0.30, "MEDIUM",   "Plan inspection this week - increase monitoring frequency"),
    (0.00, "LOW",      "Healthy - continue normal operation"),
]

# Business assumptions used by reports/business_impact.py (industry-typical values)
DOWNTIME_COST_PER_HOUR       = 4500.0  # USD/h of unplanned production loss
AVG_UNPLANNED_DOWNTIME_HOURS = 8.0     # typical unplanned stop length
PLANNED_MAINTENANCE_COST     = 400.0   # USD per scheduled intervention (crew + parts)
INSPECTION_COST              = 60.0    # USD for triaging one AI alarm
FALSE_ALARM_STOP_HOURS       = 0.5     # line paused while crew checks a false alarm
SENSOR_KIT_COST              = 250.0   # USD per instrumented machine (edge retrofit)

# Derived decision economics (used by the threshold optimiser)
MISSED_FAILURE_COST = AVG_UNPLANNED_DOWNTIME_HOURS * DOWNTIME_COST_PER_HOUR      # 36,000
FALSE_ALARM_COST    = INSPECTION_COST + FALSE_ALARM_STOP_HOURS * DOWNTIME_COST_PER_HOUR  # 2,310
THRESHOLD_MODE      = "cost"   # 'cost' -> minimise money lost, 'f1' -> balance P/R

# Plant scenario used by reports (kept separate from the datasets so we never
# pretend a benchmark's failure density equals a real factory's failure rate)
BREAKDOWNS_PER_MACHINE_YEAR = 2.0   # industry-typical for monitored rotating equipment
CONVERSION_SCENARIOS = {            # % of AI alarms actually acted on in time
    "conservative": 0.40, "expected": 0.70, "optimistic": 0.90,
}

# ---------------------------------------------------------------------------
# 7. MODEL / ARTEFACT NAMES
# ---------------------------------------------------------------------------
BEST_MODEL_PATH  = MODELS_DIR / "predictive_maintenance_model.pkl"
AI4I_METRICS     = MODELS_DIR / "ai4i_metrics.json"
STREAM_MODEL_PATH = MODELS_DIR / "stream_failure_model.pkl"
STREAM_METRICS   = MODELS_DIR / "stream_metrics.json"
MODEL_CARD       = MODELS_DIR / "model_card.json"

PREDICTIONS_CSV  = OUTPUTS_DIR / "test_predictions.csv"
LIVE_ALERTS_CSV  = OUTPUTS_DIR / "live_alerts.csv"
STREAM_PRED_CSV  = OUTPUTS_DIR / "stream_predictions.csv"
DASHBOARD_HTML   = OUTPUTS_DIR / "dashboard.html"
BUSINESS_REPORT  = REPORTS_DIR / "business_impact.md"
MODEL_COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"

# Plot style
PLOT_DPI = 130
