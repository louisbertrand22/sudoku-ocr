# stub cells
import cv2
import numpy as np

def _peaks_from_projection(proj: np.ndarray, n: int, min_gap: int = 2):
    """Regroupe les colonnes/lignes contiguës 'fortes' en pics et retourne leurs centres.
       Si > n, prend les n plus denses; si < n, fallback linéaire."""
    thresh = 0.5 * float(proj.max())
    idx = np.where(proj > thresh)[0]
    if idx.size == 0:
        return None
    # groupes de colonnes/lignes contiguës
    splits = np.where(np.diff(idx) > min_gap)[0] + 1
    groups = np.split(idx, splits)
    # score = somme du proj dans le groupe, centre = moyenne indices
    scored = []
    for g in groups:
        if g.size == 0:
            continue
        score = float(proj[g].sum())
        center = int(round(g.mean()))
        scored.append((score, center, g[0], g[-1]))
    # tri par score décroissant et prise des n meilleurs
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:n]
    centers = sorted([c for _, c, _, _ in top])
    if len(centers) != n:
        return None
    return centers

def detect_grid_lines(warped: np.ndarray) -> tuple[list[int], list[int]]:
    """Retourne (xs, ys): positions des 10 lignes verticales et 10 horizontales (dans l’image rectifiée)."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 2
    )
    H, W = thr.shape[:2]
    # extraire les lignes
    v_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, H // 20)))
    h_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, W // 20), 1))
    v_lines = cv2.dilate(cv2.erode(thr, v_ker, 1), v_ker, 1)
    h_lines = cv2.dilate(cv2.erode(thr, h_ker, 1), h_ker, 1)

    v_proj = v_lines.sum(axis=0)  # (W,)
    h_proj = h_lines.sum(axis=1)  # (H,)
    xs = _peaks_from_projection(v_proj, n=10)
    ys = _peaks_from_projection(h_proj, n=10)

    # fallbacks si besoin : grille régulière
    if xs is None:
        xs = [int(round(i * (W - 1) / 9)) for i in range(10)]
    if ys is None:
        ys = [int(round(i * (H - 1) / 9)) for i in range(10)]
    return xs, ys

def split_cells_by_lines(warped: np.ndarray, xs: list[int], ys: list[int]) -> list:
    """Découpe les 81 cellules en s’appuyant sur les lignes détectées."""
    cells = []
    for r in range(9):
        for c in range(9):
            y1, y2 = ys[r], ys[r + 1]
            x1, x2 = xs[c], xs[c + 1]
            # petite marge pour éviter la ligne noire
            pad = max(1, (y2 - y1) // 40)
            cell = warped[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
            cells.append(cell)
    return cells

def extract_digit(cell: np.ndarray) -> np.ndarray | None:
    """Retourne une image 28x28 (digit blanc sur noir) ou None si case vide."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    # virer les bords
    h, w = thr.shape
    margin = max(2, h // 14)
    thr[:margin, :] = 0; thr[-margin:, :] = 0
    thr[:, :margin] = 0; thr[:, -margin:] = 0

    # petit nettoyage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thr, 8)
    if num_labels <= 1:
        return None

    # plus grande composante plausible (évite l’ombre/gris de fond)
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = 1 + np.argmax(areas)
    area = int(areas.max())
    cell_area = h * w
    if not (max(40, int(0.007 * cell_area)) <= area <= int(0.5 * cell_area)):
        return None

    x, y, w2, h2, _ = stats[max_idx]
    roi = (labels == max_idx).astype(np.uint8) * 255
    roi = roi[y:y + h2, x:x + w2]

    side = max(h2, w2) + 8
    canvas = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h2) // 2
    x_off = (side - w2) // 2
    canvas[y_off:y_off + h2, x_off:x_off + w2] = roi

    digit28 = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
    return digit28