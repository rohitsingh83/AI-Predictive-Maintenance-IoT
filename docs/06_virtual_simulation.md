# 6. Virtual Fleet Simulation & Physics Degradation Model

---

## 6.1 Why a Virtual IoT Fleet?

In industrial AI, raw failure data is strictly proprietary and rarely public due to liability and IP constraints. The public UCI AI4I 2020 dataset provides static cycle records, but lacks **multi-asset continuous streaming trajectories**.

To bridge this gap without requiring access to a physical multi-million-dollar factory, [`src/simulate_iot_sensors.py`](file:///c:/Users/Rohit%20Singh/OneDrive/Desktop/IITD%20PROJECTS/AI%20Projects/AI-Predictive-Maintenance-IoT/src/simulate_iot_sensors.py) implements a **first-principles physics-based degradation engine**.

---

## 6.2 Mathematical Degradation Dynamics

Each of the 30 virtual machines runs across 900 operating cycles (sampled at 10-minute intervals). Machine health decays according to an exponential degradation trajectory:

### 1. Progressive Wear Accumulation
$$H_t = H_{t-1} - \eta_{wear} - \epsilon_t$$
where $\eta_{wear}$ accelerates exponentially once micro-cracking and surface fatigue initiate:
$$\eta_{wear} = \alpha_0 \cdot \exp\left(\beta \cdot \frac{t}{T_{life}}\right)$$

### 2. Spindle Vibration Dynamics
As bearing clearance increases and cutting tool flank wear worsens, kinetic vibration velocity rises:
$$v(t) = v_{nominal} + \gamma \cdot (1 - H_t)^2 + \mathcal{N}(0, \sigma_{noise}^2)$$
When $H_t < 0.20$, high-frequency harmonic bursts emerge, triggering the predictive horizon.

### 3. Thermal Dissipation Deficit
In Heat Dissipation Failures (HDF), coolant flow restriction causes thermal accumulation:
$$T_{process}(t) = T_{air}(t) + \Delta T_{nominal} \cdot \left(1 - \lambda \cdot (1 - H_t)\right) + \frac{P_{mech}}{C_{th}}$$
When $\Delta T = T_{process} - T_{air} < 8.6\text{K}$, heat dissipation capacity is compromised, causing rapid thermal runaway.

---

## 6.3 Simulation Parameters

```python
TOTAL_MACHINES = 30
CYCLES_PER_MACHINE = 900
SAMPLING_INTERVAL = "10 minutes"
TOTAL_SAMPLES = 27,000
FAILURE_EVENTS = 118
AVERAGE_NOTICE_WINDOW = "5.3 cycles (~54 minutes)"
```

To regenerate the virtual fleet with fresh random seeds:
```bash
python main.py simulate --machines 30 --cycles 900
```
Telemetry is saved directly to [`data/raw/iot_sensor_stream.csv`](file:///c:/Users/Rohit%20Singh/OneDrive/Desktop/IITD%20PROJECTS/AI%20Projects/AI-Predictive-Maintenance-IoT/data/raw/iot_sensor_stream.csv).
