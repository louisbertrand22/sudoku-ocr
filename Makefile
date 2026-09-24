PYTHON ?= .venv/bin/python
IMAGE  ?= data/samples/sudoku4.png

.PHONY: venv install test test-fast train run ui clean

venv:  ## crée .venv (TensorFlow exige Python <= 3.13)
	uv venv -p 3.12 .venv

install:  ## installe le paquet en mode éditable + extras
	uv pip install --python $(PYTHON) -e ".[tesseract,train,ui,dev]"

test:  ## tous les tests (e2e ignorés si models/sudoku_cnn.keras absent)
	$(PYTHON) -m pytest -q

test-fast:  ## tests sans le modèle CNN
	$(PYTHON) -m pytest -q -m "not e2e"

train:  ## entraîne models/sudoku_cnn.keras (+ .meta.json) sur data/assets
	$(PYTHON) scripts/train_cnn.py --data data/assets

run:  ## résout IMAGE=... avec configs/default.yaml
	$(PYTHON) -m sudoku_ocr.cli --image $(IMAGE) --out data/outputs/result.jpg

ui:  ## interface web sur http://localhost:8501
	.venv/bin/sudoku-ocr-ui

clean:
	rm -rf build dist src/*.egg-info .pytest_cache
	find . -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
