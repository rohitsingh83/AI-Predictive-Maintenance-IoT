# ---- AI-Powered Predictive Maintenance System ----
.PHONY: help setup simulate train fast evaluate visualize report dashboard api demo test all docker clean

help:            ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:           ## create venv and install requirements
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

simulate:        ## generate the virtual IoT sensor fleet
	python main.py simulate

train:           ## train both models (full)
	python train_model.py

fast:            ## quick smoke-test training
	python train_model.py --fast

evaluate:        ## metrics + model cards
	python main.py evaluate

visualize:       ## regenerate all figures
	python main.py visualize

report:          ## business impact report
	python main.py report

dashboard:       ## build outputs/dashboard.html
	python main.py dashboard

api:             ## run the Flask inference API
	python api/app.py

demo:            ## 30-second console demo
	python main.py demo

test:            ## run the test suite
	pytest -q

all:             ## full pipeline end-to-end
	python main.py all

docker:          ## build and run with compose
	docker compose up --build

clean:           ## remove generated artefacts (keeps the AI4I raw dataset)
	rm -rf data/processed/*.csv data/raw/iot_sensor_stream.csv \
	       models/*.pkl models/*.json images/*.png \
	       outputs/*.csv outputs/*.json outputs/*.html reports/*.csv reports/*.md \
	       __pycache__ src/__pycache__ .pytest_cache
