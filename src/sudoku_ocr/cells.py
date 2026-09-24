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

BLANK_CONTRAST = 40  # vides mesurés <= 2, chiffres >= 206 (sudoku2/3.png)

def is_blank_cell(gray: np.ndarray, contrast_min: float = BLANK_CONTRAST) -> bool:
    """Case vide si son centre est quasi uniforme (fond blanc ou grisé).

    Contraste = écart entre percentiles 2 et 98 du centre de la case, robuste au
    bruit et insensible au fond grisé, contrairement au seuillage adaptatif qui y
    fait apparaître des bords.
    """
    h, w = gray.shape[:2]
    center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    if center.size == 0:
        return True
    lo, hi = np.percentile(center, (2, 98))
    return float(hi - lo) < contrast_min

def digit_from_binary(thr: np.ndarray) -> np.ndarray | None:
    """Choisit la composante "chiffre" d'une case binarisée (encre blanche sur noir).

    Retourne une image 28x28 (digit blanc sur noir) ou None si la case est vide.
    Rejette les fragments de lignes de grille et les bords de zones grisées :
    composantes trop fines, trop petites, traversant la case ou décentrées.
    """
    thr = thr.copy()
    h, w = thr.shape
    margin = max(2, h // 14)
    thr[:margin, :] = 0; thr[-margin:, :] = 0
    thr[:, :margin] = 0; thr[:, -margin:] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thr, 8)
    best, best_dist = None, None
    for i in range(1, num_labels):
        x, y, w2, h2, area = stats[i]
        if not (max(25, 0.007 * h * w) <= area <= 0.5 * h * w):
            continue
        if h2 < 0.3 * h:                  # trop petit : bruit, point
            continue
        if h2 > 0.9 * h or w2 > 0.8 * w:  # traverse la case : ligne ou bord de zone grisée
            continue
        if area / h2 < max(2.0, 0.04 * w):  # trait trop fin : fragment de ligne
            continue
        dx = (x + w2 / 2) - w / 2
        dy = (y + h2 / 2) - h / 2
        if abs(dx) > 0.25 * w or abs(dy) > 0.25 * h:  # décentré : collé à un bord
            continue
        dist = dx * dx + dy * dy
        if best is None or dist < best_dist:
            best, best_dist = i, dist
    if best is None:
        return None

    x, y, w2, h2, _ = stats[best]
    roi = (labels == best).astype(np.uint8) * 255
    roi = roi[y:y + h2, x:x + w2]

    side = max(h2, w2) + 8
    canvas = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h2) // 2
    x_off = (side - w2) // 2
    canvas[y_off:y_off + h2, x_off:x_off + w2] = roi

    return cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)

def extract_digit(cell: np.ndarray) -> np.ndarray | None:
    """Retourne une image 28x28 (digit blanc sur noir) ou None si case vide."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    if is_blank_cell(gray):
        return None
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    # petit nettoyage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    return digit_from_binary(thr)
