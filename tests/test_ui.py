import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("tensorflow")
from streamlit.testing.v1 import AppTest

from sudoku_ocr.ui.helpers import (
    bgr_to_hex, decode_image, draw_reading, encode_png, frame_to_grid, grid_to_frame, hex_to_bgr,
)

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "src" / "sudoku_ocr" / "ui" / "app.py")
SAMPLES = ["sudoku2.png", "sudoku3.png", "sudoku4.png"]
WEIGHTS = Path(os.environ.get("SUDOKU_OCR_WEIGHTS", ROOT / "models" / "sudoku_cnn.keras"))
PUZZLE4 = "205308409070000050904000607500040002000507000600030008406000801020000060801209704"


# ---------------------------------------------------------------- helpers -- #

def test_grid_frame_roundtrip():
    grid = np.array([int(c) for c in PUZZLE4]).reshape(9, 9)
    df = grid_to_frame(grid)
    assert df.isna().to_numpy().sum() == int((grid == 0).sum())  # vides -> None
    assert np.array_equal(frame_to_grid(df), grid)


def test_frame_to_grid_ignores_out_of_range_values():
    df = grid_to_frame(np.zeros((9, 9), dtype=int)).astype("object")
    df.iloc[0, 0], df.iloc[0, 1], df.iloc[0, 2], df.iloc[0, 3] = 12, -1, "x", 4.0
    grid = frame_to_grid(df)
    assert grid[0, :4].tolist() == [0, 0, 0, 4]


def test_color_conversions():
    assert hex_to_bgr("#ff8000") == [0, 128, 255]
    assert bgr_to_hex([0, 128, 255]) == "#ff8000"


def test_image_encode_decode():
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    img[5:10, 5:10] = (10, 20, 30)
    assert np.array_equal(decode_image(encode_png(img)), img)
    assert decode_image(b"") is None
    assert decode_image(b"pas une image") is None


def test_draw_reading_highlights_conflicts():
    warped = np.full((450, 450, 3), 255, dtype=np.uint8)
    lines = [round(i * 449 / 9) for i in range(10)]
    out = draw_reading(warped, lines, lines, np.zeros((9, 9), dtype=int), [(0, 0)])
    assert out[25, 25, 2] > out[25, 25, 0]        # case (0,0) teintée en rouge
    assert tuple(out[274, 274]) == (255, 255, 255)  # centre d'une autre case, intact


# ------------------------------------------------------------------- app -- #

def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120)


def _editor_key(name: str) -> str:
    image_id = hashlib.sha1((ROOT / "data" / "samples" / name).read_bytes()).hexdigest()[:12]
    return f"editor-{image_id}-0"


def _edit(at: AppTest, name: str, edited_rows: dict) -> None:
    at.session_state[_editor_key(name)] = {"edited_rows": edited_rows, "added_rows": [],
                                           "deleted_rows": []}
    at.run()


def test_missing_model_shows_training_command(monkeypatch):
    monkeypatch.chdir(ROOT)
    at = _app().run()
    at.sidebar.text_input[0].input("models/absent.keras").run()
    assert not at.exception
    assert any("Modèle introuvable" in e.value for e in at.error)
    assert any("train_cnn.py" in c.value for c in at.code)


@pytest.fixture
def at(monkeypatch):
    if not WEIGHTS.exists():
        pytest.skip(f"modèle absent : {WEIGHTS}")
    monkeypatch.chdir(ROOT)
    app = _app().run()
    # chaque exécution d'AppTest coûte ~6 s (cache vidé) : ne relancer que si
    # WEIGHTS n'est pas déjà le modèle de configs/default.yaml
    if Path(app.sidebar.text_input[0].value).resolve() != WEIGHTS.resolve():
        app.sidebar.text_input[0].input(str(WEIGHTS)).run()
    return app


@pytest.mark.e2e
@pytest.mark.parametrize("name", SAMPLES)
def test_samples_are_solved(at, name):
    at.selectbox[0].select(Path("data/samples") / name).run()
    assert not at.exception
    assert [s.value for s in at.success] == ["Grille valide, solution unique."]
    assert len(at.get("download_button")) == 1


@pytest.mark.e2e
def test_conflict_is_reported_and_blocks_solution(at):
    at.selectbox[0].select(Path("data/samples") / "sudoku4.png").run()
    _edit(at, "sudoku4.png", {0: {"2": 2}})  # 2 déjà en L1C1 et L8C2
    assert any("L1C1, L1C2, L8C2" in e.value for e in at.error)
    assert len(at.get("download_button")) == 0


@pytest.mark.e2e
def test_missing_digits_give_multiple_solutions_warning(at):
    at.selectbox[0].select(Path("data/samples") / "sudoku4.png").run()
    _edit(at, "sudoku4.png", {0: {c: None for c in "134679"}, 1: {"2": None, "8": None},
                              2: {c: None for c in "1379"}})
    assert any("Plusieurs solutions" in w.value for w in at.warning)
    assert at.metric[0].value == "20"
    assert len(at.get("download_button")) == 0


@pytest.mark.e2e
def test_undo_restores_ocr_reading(at):
    at.selectbox[0].select(Path("data/samples") / "sudoku4.png").run()
    _edit(at, "sudoku4.png", {0: {"2": 2}})
    at.button[0].click().run()
    assert [s.value for s in at.success] == ["Grille valide, solution unique."]


@pytest.mark.e2e
def test_uploaded_image_without_grid_shows_error(at):
    at.segmented_control[0].set_value("Importer une image").run()
    assert any("Importez une image" in i.value for i in at.info)
