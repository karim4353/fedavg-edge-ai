# Convenience targets for the FedAvg Edge AI project
.PHONY: install test smoke run all clean help

PY ?= python

help:
	@echo "Targets:"
	@echo "  install   - install dependencies"
	@echo "  test      - run the test suite"
	@echo "  smoke     - run a tiny end-to-end experiment (CI-friendly)"
	@echo "  run       - run the full experiment suite (default config)"
	@echo "  all       - install, test, smoke, run"
	@echo "  clean     - remove generated artifacts"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

smoke:
	$(PY) -m experiments.run_all --config configs/smoke.json

run:
	$(PY) -m experiments.run_all --config configs/default.json

all: install test smoke run

clean:
	rm -rf reports/plots/*.png reports/tables/*.csv reports/*.md
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
