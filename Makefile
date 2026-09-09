.PHONY: install test inspect build build-dense infer submission tune

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

inspect:
	python scripts/inspect_parser.py --source data/selected-contexts.zip --limit 3

build:
	python scripts/build_index.py --config configs/default.yaml

build-dense:
	python scripts/build_index.py --config configs/default.yaml --dense-only

infer:
	python scripts/infer.py --config configs/default.yaml --input data/public-official.json --output runs/default/public_predictions.json

submission:
	powershell -ExecutionPolicy Bypass -File create_submission.ps1 -Fresh

tune:
	python scripts/tune_threshold.py --config configs/default.yaml --val-ratio 0.15
