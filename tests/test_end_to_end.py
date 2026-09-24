"""Tests de bout en bout sur les images d'exemple, avec le vrai modèle CNN.

Le modèle n'est pas versionné : ces tests utilisent SUDOKU_OCR_WEIGHTS, sinon
models/sudoku_cnn.keras, et sont ignorés s'il n'existe pas. Pour le produire :
    python scripts/train_cnn.py --data data/assets
Pour les exclure : pytest -m "not e2e"
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from sudoku_ocr import cli
from sudoku_ocr.pipeline import read_grid, run
from sudoku_ocr.solver import solved_ok

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
WEIGHTS = Path(os.environ.get("SUDOKU_OCR_WEIGHTS", ROOT / "models" / "sudoku_cnn.keras"))

# Grilles attendues, ligne par ligne (0 = case vide)
EXPECTED = {
    "sudoku2.png": "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
    "sudoku3.png": "543070619102069703607000004250080970074205800801006540708641300305928467400037120",
    "sudoku4.png": "205308409070000050904000607500040002000507000600030008406000801020000060801209704",
}


def _expected(name: str) -> np.ndarray:
    return np.array([int(c) for c in EXPECTED[name]]).reshape(9, 9)


def _describe_diff(got: np.ndarray, want: np.ndarray) -> str:
    lines = []
    for r, c in zip(*np.nonzero(got != want)):
        kind = "faux positif" if want[r, c] == 0 else "manqué" if got[r, c] == 0 else "mal lu"
        lines.append(f"  ({r},{c}) lu {got[r, c]} attendu {want[r, c]} : {kind}")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def cfg():
    if not WEIGHTS.exists():
        pytest.skip(f"modèle absent : {WEIGHTS} (python scripts/train_cnn.py --data data/assets)")
    return {"ocr": {"backend": "cnn", "cnn_weights": str(WEIGHTS)}}


@pytest.fixture(scope="module")
def ocr(cfg):
    from sudoku_ocr.pipeline import _make_ocr_backend
    return _make_ocr_backend("cnn", cfg)  # chargé une seule fois pour le module


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_reads_every_given_digit(name, cfg, ocr):
    img = cv2.imread(str(SAMPLES / name))
    grid = read_grid(img, cfg, ocr).grid
    want = _expected(name)
    assert np.array_equal(grid, want), f"{name} :\n{_describe_diff(grid, want)}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_run_writes_a_valid_solution(name, cfg, ocr, tmp_path):
    out = tmp_path / f"result_{name}"
    res = run(str(SAMPLES / name), str(out), cfg, ocr=ocr)

    solution = np.array(res["solution"])
    want = _expected(name)
    assert solved_ok(solution)
    assert np.array_equal(solution[want != 0], want[want != 0]), "la solution modifie des chiffres de départ"

    written = cv2.imread(str(out))
    assert written is not None
    assert written.shape == cv2.imread(str(SAMPLES / name)).shape


def test_cli_solves_sample(cfg, tmp_path, capsys):
    out = tmp_path / "cli.jpg"
    code = cli.main(["--image", str(SAMPLES / "sudoku4.png"), "--weights", cfg["ocr"]["cnn_weights"],
                     "--out", str(out)])
    assert code == 0, capsys.readouterr().err
    assert out.exists()
