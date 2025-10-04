# stub geometry
import cv2
import numpy as np


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """
    Ordonne 4 points d'un quadrilatère au format [tl, tr, br, bl]
    (top-left, top-right, bottom-right, bottom-left).
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)           # tl: min(s), br: max(s)
    diff = np.diff(pts, axis=1)   # tr: min(diff), bl: max(diff)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def four_point_transform(image: np.ndarray, quad: np.ndarray, size: int = 450):
    """
    Applique une homographie pour redresser la grille en un carré size×size.
    Retourne (warped, M, Minv) où M est la matrice de perspective et Minv son inverse.
    """
    ordered = _order_quad(quad)
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (size, size))

    Minv = cv2.getPerspectiveTransform(dst, ordered)
    return warped, M, Minv
