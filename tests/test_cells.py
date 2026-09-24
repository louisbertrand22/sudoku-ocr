import cv2
import numpy as np

from sudoku_ocr.cells import extract_digit, is_blank_cell

SIZE = 48


def _cell(bg: int = 255) -> np.ndarray:
    return np.full((SIZE, SIZE, 3), bg, dtype=np.uint8)


def _with_digit(cell: np.ndarray, text: str = "1") -> np.ndarray:
    cv2.putText(cell, text, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    return cell


def test_white_cell_is_blank():
    assert extract_digit(_cell(255)) is None


def test_shaded_cell_is_blank():
    assert extract_digit(_cell(215)) is None


def test_shaded_cell_with_inner_edges_is_blank():
    # bord de zone grisée décalé de quelques pixels : cadre clair dans une case grise
    cell = _cell(215)
    cell[:6, :] = 255
    cell[:, :6] = 255
    assert extract_digit(cell) is None


def test_grid_line_fragment_is_blank():
    # ligne de grille mal rognée, collée au bord de la case
    cell = _cell(255)
    cell[4:44, 5:7] = 0
    assert extract_digit(cell) is None


def test_digit_one_is_detected():
    d = extract_digit(_with_digit(_cell(255), "1"))
    assert d is not None and d.shape == (28, 28)


def test_digit_on_shaded_cell_is_detected():
    assert extract_digit(_with_digit(_cell(215), "7")) is not None


def test_digit_next_to_grid_line_is_detected():
    cell = _with_digit(_cell(255), "4")
    cell[:, 5:7] = 0
    assert extract_digit(cell) is not None


def test_is_blank_cell_uses_center_contrast():
    gray = np.full((SIZE, SIZE), 200, dtype=np.uint8)
    assert is_blank_cell(gray)
    gray[20:28, 22:26] = 0
    assert not is_blank_cell(gray)
