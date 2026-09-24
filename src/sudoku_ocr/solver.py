from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

Grid = np.ndarray  # shape (9,9), dtype=int


def find_empty(grid: Grid) -> Optional[Tuple[int, int]]:
    """Retourne la première case vide (valeur 0) ou None si plein."""
    pos = np.argwhere(grid == 0)
    if pos.size == 0:
        return None
    r, c = pos[0]
    return int(r), int(c)


def _row_ok(grid: Grid, r: int, v: int) -> bool:
    return v not in grid[r, :]


def _col_ok(grid: Grid, c: int, v: int) -> bool:
    return v not in grid[:, c]


def _box_ok(grid: Grid, r: int, c: int, v: int) -> bool:
    br, bc = (r // 3) * 3, (c // 3) * 3
    return v not in grid[br:br+3, bc:bc+3]


def is_valid(grid: Grid, r: int, c: int, v: int) -> bool:
    """Teste si on peut placer v en (r,c)."""
    if v == 0:
        return True
    return _row_ok(grid, r, v) and _col_ok(grid, c, v) and _box_ok(grid, r, c, v)


def _candidates(grid: Grid, r: int, c: int):
    """Ensemble des candidats possibles pour la case (r,c)."""
    if grid[r, c] != 0:
        return []
    used = set(grid[r, :]) | set(grid[:, c]) | set(grid[(r//3)*3:(r//3)*3+3, (c//3)*3:(c//3)*3+3].ravel())
    return [v for v in range(1, 10) if v not in used]


def solve(grid: Grid) -> bool:
    """
    Résout la grille en place avec backtracking. Retourne True si succès.
    Algorithme :
      - Heuristique MRV (Minimum Remaining Values) : on choisit la case vide
        avec le moins de candidats pour couper rapidement l'espace de recherche.
    """
    # Trouver toutes les cases vides et choisir celle avec le moins de candidats
    empties = [(r, c) for r in range(9) for c in range(9) if grid[r, c] == 0]
    if not empties:
        return True

    # Calculer les candidats et trier par nombre croissant
    candidates_list = []
    for (r, c) in empties:
        cand = _candidates(grid, r, c)
        if not cand:
            return False  # contradiction
        candidates_list.append((len(cand), r, c, cand))
    candidates_list.sort(key=lambda t: t[0])

    _, r, c, cand = candidates_list[0]
    for v in cand:
        if is_valid(grid, r, c, v):
            grid[r, c] = v
            if solve(grid):
                return True
            grid[r, c] = 0
    return False


def solved_ok(grid: Grid) -> bool:
    """Vérifie qu'une grille complète est valide (chaque ligne/col/box = 1..9)."""
    if np.any(grid == 0):
        return False
    target = set(range(1, 10))
    for i in range(9):
        if set(grid[i, :]) != target:
            return False
        if set(grid[:, i]) != target:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            if set(grid[br:br+3, bc:bc+3].ravel()) != target:
                return False
    return True


def find_conflicts(grid: Grid) -> list[Tuple[int, int]]:
    """Cases (r, c) dont la valeur est répétée dans sa ligne, sa colonne ou sa boîte."""
    conflicts = set()
    units = [[(r, c) for c in range(9)] for r in range(9)]
    units += [[(r, c) for r in range(9)] for c in range(9)]
    units += [[(br + i, bc + j) for i in range(3) for j in range(3)]
              for br in range(0, 9, 3) for bc in range(0, 9, 3)]
    for unit in units:
        seen: dict[int, list[Tuple[int, int]]] = {}
        for r, c in unit:
            v = int(grid[r, c])
            if v:
                seen.setdefault(v, []).append((r, c))
        for cells in seen.values():
            if len(cells) > 1:
                conflicts.update(cells)
    return sorted(conflicts)


def count_solutions(grid: Grid, limit: int = 2) -> int:
    """Nombre de solutions, plafonné à `limit` (2 suffit pour tester l'unicité).

    Ne modifie pas `grid`. Une grille avec des conflits a 0 solution.
    """
    if find_conflicts(grid):
        return 0
    work = np.array(grid, dtype=int, copy=True)

    def _count() -> int:
        empties = [(r, c) for r in range(9) for c in range(9) if work[r, c] == 0]
        if not empties:
            return 1
        best = None
        for r, c in empties:
            cand = _candidates(work, r, c)
            if not cand:
                return 0
            if best is None or len(cand) < len(best[2]):
                best = (r, c, cand)
        r, c, cand = best
        total = 0
        for v in cand:
            work[r, c] = v
            total += _count()
            work[r, c] = 0
            if total >= limit:
                break
        return total

    return min(_count(), limit)
