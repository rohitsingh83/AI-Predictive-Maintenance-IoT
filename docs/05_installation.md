# 5. Installation & Environment Setup

---

## 5.1 System Prerequisites

* **Operating System:** Windows 10/11, macOS 12+, or Ubuntu 20.04+
* **Python Runtime:** Python 3.9, 3.10, or 3.11 (Python 3.11 recommended)
* **RAM / CPU:** 4 GB RAM minimum; 2 CPU cores (no GPU required)
* **Disk Space:** ~500 MB for environment, data, and model bundles

---

## 5.2 Windows Setup (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/rohitsingh83/AI-Predictive-Maintenance-IoT.git
cd AI-Predictive-Maintenance-IoT

# 2. Allow script execution in current PowerShell session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the entire pipeline (simulation, training, evaluation, figures, dashboard)
python main.py all
```

---

## 5.3 macOS & Linux Setup (Bash / Zsh)

```bash
# 1. Clone the repository
git clone https://github.com/rohitsingh83/AI-Predictive-Maintenance-IoT.git
cd AI-Predictive-Maintenance-IoT

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the entire pipeline
python main.py all
```

---

## 5.4 Docker Deployment

To launch the system as an isolated container:

```bash
# Build the Docker image
docker build -t pdm-iot:latest .

# Run the REST API container on port 5000
docker run -d -p 5000:5000 --name pdm-api pdm-iot:latest

# Or launch via Docker Compose
docker compose up -d
```
Verify the container health at `http://localhost:5000/health`.
