import numpy as np
import pytest

from sudoku_ocr.solver import find_empty, is_valid, solve, solved_ok


def _grid(s: str) -> np.ndarray:
    return np.array([int(c) for c in s]).reshape(9, 9)


# grille de data/samples/sudoku4.png et sa solution
PUZZLE = "205308409070000050904000607500040002000507000600030008406000801020000060801209704"
SOLUTION = "265378419178496253934125687589641372342587196617932548496753821723814965851269734"


def test_solves_known_puzzle():
    grid = _grid(PUZZLE)
    assert solve(grid)
    assert np.array_equal(grid, _grid(SOLUTION))


def test_solution_keeps_givens_and_is_valid():
    grid = _grid(PUZZLE)
    solve(grid)
    givens = _grid(PUZZLE) != 0
    assert np.array_equal(grid[givens], _grid(PUZZLE)[givens])
    assert solved_ok(grid)


def test_solves_empty_grid():
    grid = np.zeros((9, 9), dtype=int)
    assert solve(grid)
    assert solved_ok(grid)


def test_cell_without_candidate_is_unsolvable():
    grid = np.zeros((9, 9), dtype=int)
    grid[0, :8] = [1, 2, 3, 4, 5, 6, 7, 8]  # (0,8) ne peut être que 9...
    grid[5, 8] = 9                          # ... mais 9 est déjà dans la colonne
    assert not solve(grid)


@pytest.mark.parametrize("r, c, v, ok", [
    (0, 1, 6, True),    # valeur de la solution
    (0, 1, 2, False),   # déjà dans la ligne
    (0, 1, 7, False),   # déjà dans la colonne
    (0, 1, 4, False),   # déjà dans la ligne (4 en (0,6))
    (1, 0, 9, False),   # déjà dans la boîte (9 en (2,0))
    (0, 1, 0, True),    # 0 = vider une case, toujours autorisé
])
def test_is_valid(r, c, v, ok):
    assert is_valid(_grid(PUZZLE), r, c, v) is ok


def test_find_empty():
    assert find_empty(_grid(PUZZLE)) == (0, 1)
    assert find_empty(_grid(SOLUTION)) is None


def test_solved_ok_rejects_invalid_grids():
    bad = _grid(SOLUTION)
    bad[0, 0], bad[0, 1] = bad[0, 1], bad[0, 0]  # permutation : colonnes/boîtes fausses
    assert not solved_ok(bad)
    incomplete = _grid(SOLUTION)
    incomplete[4, 4] = 0
    assert not solved_ok(incomplete)
