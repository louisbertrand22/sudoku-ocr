from pathlib import Path

import cv2
import numpy as np
import pytest

from sudoku_ocr.cells import detect_grid_lines, grid_score
from sudoku_ocr.detect import find_sudoku_quad
from sudoku_ocr.geometry import _order_quad, four_point_transform

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"


def _draw_grid(img, x0, y0, size):
    step = size / 9
    for i in range(10):
        t = 5 if i % 3 == 0 else 1
        p = int(round(i * step))
        cv2.line(img, (x0 + p, y0), (x0 + p, y0 + size), (0, 0, 0), t)
        cv2.line(img, (x0, y0 + p), (x0 + size, y0 + p), (0, 0, 0), t)


def _assert_corners(quad, expected, tol):
    got = _order_quad(quad)
    assert np.abs(got - np.array(expected, dtype=np.float32)).max() <= tol, got.tolist()


def test_grid_inside_frame_is_preferred_over_frame():
    img = np.full((600, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (597, 597), (0, 0, 0), 2)  # cadre fin autour de l'image
    _draw_grid(img, 120, 100, 360)
    quad = find_sudoku_quad(img)
    _assert_corners(quad, [[120, 100], [480, 100], [480, 460], [120, 460]], tol=6)


def test_grid_filling_image_is_detected():
    img = np.full((460, 460, 3), 255, dtype=np.uint8)
    _draw_grid(img, 5, 5, 450)
    quad = find_sudoku_quad(img)
    _assert_corners(quad, [[5, 5], [455, 5], [455, 455], [5, 455]], tol=6)


def test_grid_score_separates_grid_from_blank():
    img = np.full((450, 450, 3), 255, dtype=np.uint8)
    assert grid_score(img) == 0.0
    _draw_grid(img, 0, 0, 449)
    assert grid_score(img) > 0.9


@pytest.mark.parametrize("name", ["sudoku2.png", "sudoku3.png", "sudoku4.png"])
def test_sample_images_give_regular_grid(name):
    img = cv2.imread(str(SAMPLES / name))
    quad = find_sudoku_quad(img)
    warped, _, _ = four_point_transform(img, quad, size=450)
    xs, ys = detect_grid_lines(warped)
    expected = [i * 449 / 9 for i in range(10)]
    assert np.abs(np.array(xs) - expected).max() <= 8, xs
    assert np.abs(np.array(ys) - expected).max() <= 8, ys


def test_sudoku4_ignores_image_border():
    img = cv2.imread(str(SAMPLES / "sudoku4.png"))
    quad = find_sudoku_quad(img)
    _assert_corners(quad, [[133, 120], [2319, 120], [2319, 2306], [133, 2306]], tol=15)


class _NoOCR:
    """OCR factice : la polarité se décide avant toute lecture de chiffre."""
    def predict_digit(self, img28):
        return 0


@pytest.mark.parametrize("name", ["sudoku2.png", "sudoku3.png", "sudoku4.png",
                                  "sudoku2_dark.png", "sudoku3_dark.png", "sudoku4_dark.png"])
def test_read_grid_picks_polarity(name):
    from sudoku_ocr.pipeline import read_grid
    img = cv2.imread(str(SAMPLES / name))
    reading = read_grid(img, None, ocr=_NoOCR())
    assert reading.inverted == name.endswith("_dark.png")
    # la grille affichée garde la polarité d'origine
    median = np.median(cv2.cvtColor(reading.warped, cv2.COLOR_BGR2GRAY))
    assert (median < 128) == reading.inverted
