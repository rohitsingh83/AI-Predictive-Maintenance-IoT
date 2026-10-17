"""
capture_screenshots.py
======================
Automated high-resolution screenshot generator for AI Predictive Maintenance IoT.
Uses Selenium Chrome headless to capture the live interactive platform views.
"""

import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS_DIR = ROOT / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

HTML_PATH = (ROOT / "docs" / "index.html").resolve()
DASHBOARD_PATH = (ROOT / "outputs" / "dashboard.html").resolve()

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1440,1080")
opts.add_argument("--disable-gpu")
opts.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=opts)

try:
    print(f"Loading local web platform from: {HTML_PATH.as_uri()}")
    driver.get(HTML_PATH.as_uri())
    time.sleep(2)

    # 1. Hero & KPIs
    print("[1/8] Capturing 01_website_hero_and_kpis.png...")
    driver.set_window_size(1440, 960)
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "01_website_hero_and_kpis.png"))

    # 2. Fleet Floor Matrix
    print("[2/8] Capturing 02_fleet_floor_matrix.png...")
    driver.execute_script("document.getElementById('fleet').scrollIntoView();")
    time.sleep(1)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "02_fleet_floor_matrix.png"))

    # 3. Sensor Studio (Nominal)
    print("[3/8] Capturing 03_sensor_studio_normal.png...")
    driver.execute_script("document.getElementById('studio').scrollIntoView();")
    time.sleep(1)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "03_sensor_studio_normal.png"))

    # 4. Sensor Studio (Critical HDF Alert + Work Order)
    print("[4/8] Capturing 04_sensor_studio_critical_failure.png...")
    # Click Heat Dissipation Failure preset
    preset_btns = driver.find_elements(By.CLASS_NAME, "preset-btn")
    for btn in preset_btns:
        if "Heat Dissipation" in btn.text:
            btn.click()
            break
    time.sleep(1.5)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "04_sensor_studio_critical_failure.png"))

    # 5. Degradation Trajectory
    print("[5/8] Capturing 05_degradation_trajectory.png...")
    driver.execute_script("document.getElementById('physics').scrollIntoView();")
    time.sleep(1)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "05_degradation_trajectory.png"))

    # 6. Model Benchmarks
    print("[6/8] Capturing 06_model_benchmarks_and_threshold.png...")
    driver.execute_script("document.getElementById('benchmarks').scrollIntoView();")
    time.sleep(1)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "06_model_benchmarks_and_threshold.png"))

    # 7. Plant ROI Engine
    print("[7/8] Capturing 07_plant_roi_calculator.png...")
    driver.execute_script("document.getElementById('roi').scrollIntoView();")
    time.sleep(1)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "07_plant_roi_calculator.png"))

    # 8. Operator Console Dashboard
    if DASHBOARD_PATH.exists():
        print("[8/8] Capturing 08_operator_dashboard.png...")
        driver.get(DASHBOARD_PATH.as_uri())
        time.sleep(2)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "08_operator_dashboard.png"))

    print(f"\nAll screenshots successfully saved to: {SCREENSHOTS_DIR}")

finally:
    driver.quit()
