"""Fonctions pures utilisées par l'interface (sans dépendance à Streamlit)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

COLUMNS = [str(i) for i in range(1, 10)]
INDEX = [str(i) for i in range(1, 10)]


def decode_image(data: bytes) -> np.ndarray | None:
    """Octets d'un fichier image -> image BGR, ou None si illisible."""
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Échec de l'encodage PNG")
    return buf.tobytes()


def hex_to_bgr(color: str) -> list[int]:
    """'#rrggbb' -> [b, g, r]."""
    color = color.lstrip("#")
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return [b, g, r]


def bgr_to_hex(bgr) -> str:
    b, g, r = (int(v) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def grid_to_frame(grid: np.ndarray) -> pd.DataFrame:
    """Grille 9x9 (0 = vide) -> DataFrame éditable, cases vides à None."""
    df = pd.DataFrame(np.asarray(grid, dtype=int), index=INDEX, columns=COLUMNS).astype("Int64")
    return df.mask(df == 0)


def frame_to_grid(df: pd.DataFrame) -> np.ndarray:
    """DataFrame éditée -> grille 9x9 int ; vide, hors 1..9 ou non numérique -> 0."""
    values = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, na_value=0.0)
    values = np.nan_to_num(values, nan=0.0)
    grid = values.round().astype(int)
    grid[(grid < 1) | (grid > 9)] = 0
    if grid.shape != (9, 9):
        raise ValueError(f"grille de forme {grid.shape}, attendu (9, 9)")
    return grid


def draw_detection(img_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Contour de la grille détectée sur l'image d'origine."""
    out = img_bgr.copy()
    thickness = max(2, round(max(out.shape[:2]) / 250))
    pts = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, (40, 180, 40), thickness, cv2.LINE_AA)
    for x, y in pts.reshape(-1, 2):
        cv2.circle(out, (int(x), int(y)), thickness * 3, (40, 180, 40), -1, cv2.LINE_AA)
    return out


def draw_reading(warped_bgr: np.ndarray, xs: list[int], ys: list[int],
                 grid: np.ndarray, highlight: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Grille redressée + lignes détectées + chiffres lus (coin haut gauche de chaque case).

    Les cases de `highlight` (conflits) sont teintées en rouge.
    """
    out = warped_bgr.copy()
    for r, c in highlight or []:
        tint = out[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
        tint[:] = (0.55 * tint + 0.45 * np.array([60, 60, 230])).astype(np.uint8)
    size = out.shape[0]
    for x in xs:
        cv2.line(out, (x, 0), (x, size - 1), (230, 140, 30), 1, cv2.LINE_AA)
    for y in ys:
        cv2.line(out, (0, y), (size - 1, y), (230, 140, 30), 1, cv2.LINE_AA)
    scale = size / 900
    for r in range(9):
        for c in range(9):
            v = int(grid[r, c])
            if v:
                org = (xs[c] + max(2, round(4 * scale * 2)), ys[r] + max(10, round(30 * scale)))
                cv2.putText(out, str(v), org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale * 2,
                            (230, 140, 30), max(1, round(2 * scale)), cv2.LINE_AA)
    return out
