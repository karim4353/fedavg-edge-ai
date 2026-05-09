.PHONY: install test smoke experiment full clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

smoke:
	python experiments/run_smoke.py

experiment:
	python experiments/run_full.py

full: install test smoke experiment

clean:
	rm -rf results/ reports/*.png reports/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} +
