# ---- AI-Powered Predictive Maintenance - inference API ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps for matplotlib wheels (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# generate the virtual fleet + train the models at build time so the image is
# instantly usable (skip with --build-arg SKIP_TRAIN=1 and mount your own models)
ARG SKIP_TRAIN=0
RUN if [ "$SKIP_TRAIN" = "0" ]; then python main.py simulate && python train_model.py --fast; fi

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:5000/health || exit 1

CMD ["python", "api/app.py", "--host", "0.0.0.0", "--port", "5000"]
