# stub overlay
from __future__ import annotations
import cv2
import numpy as np


def overlay_solution(
    orig_bgr: np.ndarray,
    Minv: np.ndarray,
    pred_grid: np.ndarray,
    given_mask: np.ndarray,
    color: tuple[int,int,int] = (120,120,120),       # couleur des cases résolues (non-givens)
    given_color: tuple[int,int,int] = (120,120,120),  # couleur des givens (gris)
    scale: float = 1.2,
    thickness: int = 2,
    warp_size: int = 450,
    xs: list[int] | None = None,
    ys: list[int] | None = None,
    show_mode: str = "all",  # "all" | "new" | "givens_only"
) -> np.ndarray:
    """
    show_mode:
      - "all": écrit givens (gris) + nouvelles valeurs (vert)
      - "new": n'écrit que les non-givens (comportement d'avant)
      - "givens_only": n'écrit que les givens
    """
    h_out, w_out = orig_bgr.shape[:2]
    if xs is None or ys is None:
        xs = [int(round(i*(warp_size-1)/9)) for i in range(10)]
        ys = [int(round(i*(warp_size-1)/9)) for i in range(10)]

    overlay = np.zeros((warp_size, warp_size, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_digit(text, cx, cy, col):
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
        org = (int(cx - tw/2), int(cy + th/2 - base/2))
        # contour noir puis remplissage
        cv2.putText(overlay, text, org, font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(overlay, text, org, font, scale, col, thickness, cv2.LINE_AA)

    for r in range(9):
        for c in range(9):
            v = int(pred_grid[r, c])
            if v <= 0:
                continue
            cx = (xs[c] + xs[c+1]) // 2
            cy = (ys[r] + ys[r+1]) // 2

            is_given = bool(given_mask[r, c])
            if show_mode == "new" and is_given:
                continue
            if show_mode == "givens_only" and not is_given:
                continue

            draw_digit(str(v), cx, cy, given_color if is_given else color)

    warped_back = cv2.warpPerspective(overlay, Minv, (w_out, h_out))
    gray = cv2.cvtColor(warped_back, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    inv = cv2.bitwise_not(mask)
    base = cv2.bitwise_and(orig_bgr, orig_bgr, mask=inv)
    added = cv2.bitwise_and(warped_back, warped_back, mask=mask)
    return cv2.add(base, added)