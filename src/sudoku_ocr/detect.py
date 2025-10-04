import os
import cv2
import numpy as np




def _dump_debug(name: str, img):
    """If env SUDOKU_DEBUG is set, write intermediates to SUDOKU_DEBUG_DIR (or data/outputs/debug)."""
    if not os.environ.get("SUDOKU_DEBUG"):
        return
    out_dir = os.environ.get("SUDOKU_DEBUG_DIR", "data/outputs/debug")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    cv2.imwrite(path, img)




def _approx_quad(cnt, max_eps_frac: float = 0.1):
    """Try multiple epsilon values to get a 4-point polygon. Fallback to convex hull or minAreaRect box."""
    peri = cv2.arcLength(cnt, True)
    for frac in np.linspace(0.01, max_eps_frac, 15):
        approx = cv2.approxPolyDP(cnt, frac * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    # Fallback 1: hull then approx
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for frac in np.linspace(0.01, max_eps_frac, 15):
        approx = cv2.approxPolyDP(hull, frac * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    # Fallback 2: min area rect box
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)




def find_sudoku_quad(img_bgr: np.ndarray):
    """
    Détecte la grille Sudoku et retourne ses 4 coins (float32) dans l'image d'origine.
    Stratégie robuste : CLAHE → blur → adaptive threshold → morpho close/open,
    puis plus grand contour, approximation poly (multi-epsilon), hull & box en fallback.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None


    h, w = img_bgr.shape[:2]


    # Prétraitement
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)


    thr = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 2,
    )


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)


    _dump_debug("01_gray.png", gray)
    _dump_debug("02_thr.png", thr)
    _dump_debug("03_closed.png", closed)


    # Contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None


    contours = sorted(contours, key=cv2.contourArea, reverse=True)


    # Heuristic: skip tiny contours (< 10% of image area)
    min_area = 0.1 * (h * w)
    for cnt in contours[:20]:
        if cv2.contourArea(cnt) < min_area:
            continue
        quad = _approx_quad(cnt)
        if quad is not None and len(quad) == 4:
            _dump_debug("04_contour.png", cv2.drawContours(img_bgr.copy(), [cnt], -1, (0, 255, 0), 3))
            return quad.astype(np.float32)


    # Last resort: largest contour regardless of area
    cnt = contours[0]
    quad = _approx_quad(cnt)
    if quad is not None and len(quad) == 4:
        _dump_debug("04_contour_fallback.png", cv2.drawContours(img_bgr.copy(), [cnt], -1, (0, 255, 0), 3))
        return quad.astype(np.float32)


    return None