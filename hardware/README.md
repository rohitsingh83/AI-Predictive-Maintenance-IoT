# Hardware (Optional) - Turning this project into a physical IoT rig

**The project does not need any hardware.** Dataset + simulation is enough for a
complete, defensible portfolio project. But if you have a Raspberry Pi, ~$30 of
components turns it into a physical demo - and that is what most students cannot
show.

## Option A - Zero hardware (recommended for most students)
```bash
python train_model.py            # models
python realtime_simulator.py --local   # a "virtual factory" streaming readings
python api/app.py                # REST API
```
You get the full pipeline, real metrics, work orders and a dashboard.

## Option B - Physical rig (best for interviews / final-year demo)
| Component | Purpose | Approx. cost |
|---|---|---|
| Raspberry Pi 4 (2 GB) or Arduino + ESP32 | edge gateway running the model | $35 |
| DHT22 | temperature / overheating | $4 |
| SW-420 | vibration / imbalance | $2 |
| ACS712 (30 A) + MCP3008 | current draw via SPI ADC | $6 |
| L298N + DC motor + small fan | the "machine" with a real load | $8 |
| ESP32 / ESP8266 | Wi-Fi uplink to the cloud | $5 |

Wiring is documented in the docstring of `raspberry_pi_sensor_node.py`.

```bash
# on the Pi
pip install adafruit-circuitpython-dht RPi.GPIO Adafruit-Blinka
python hardware/raspberry_pi_sensor_node.py --api http://<laptop-ip>:5000/predict/reading \
       --stop-on-critical
```

### What the demo proves
1. Raw telemetry is captured from **real sensors** through GPIO/SPI.
2. The same `PredictiveMaintenanceService` scores it (no code changes).
3. A **work order** is created and, at CRITICAL severity, the edge node **cuts the
   motor** - closing the loop from sensor to actuation, exactly like a PLC interlock.

## Cloud uplink (optional)
Replace the direct HTTP POST with MQTT so the gateway publishes to a broker and
the model lives in the cloud:

```bash
pip install paho-mqtt
# publish to  iot/pdm/<machine_id>/telemetry      (readings)
# subscribe to iot/pdm/<machine_id>/command        (stop / inspect commands)
```

This is the architecture used by AWS IoT Core / Azure IoT Hub / Google Cloud IoT:
**device -> broker -> stream processor -> model -> alerting -> CMMS**.
