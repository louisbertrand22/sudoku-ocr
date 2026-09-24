from __future__ import annotations
import os
from typing import Dict, Tuple
import cv2
import numpy as np

from .detect import find_sudoku_quad
from .geometry import four_point_transform
from .cells import detect_grid_lines, split_cells_by_lines, extract_digit, digit_from_binary, is_blank_cell
from .solver import solve, is_valid
from .overlay import overlay_solution
from .config import DEFAULTS, load_config

# OCR backends (importés paresseusement pour éviter les deps inutiles)

def _sanitize_grid(grid: np.ndarray) -> int:
    """Supprime les chiffres OCR en conflit (contraires aux règles). Renvoie le nombre supprimé."""
    removed = 0
    for r in range(9):
        for c in range(9):
            v = int(grid[r, c])
            if v == 0:
                continue
            grid[r, c] = 0
            if not is_valid(grid, r, c, v):
                removed += 1  # conflit -> on jette
            else:
                grid[r, c] = v
    return removed

def _fill_singles(grid: np.ndarray) -> int:
    """Remplit itérativement les cases à candidat unique (naked singles)."""
    filled = 0
    changed = True
    while changed:
        changed = False
        for r in range(9):
            for c in range(9):
                if grid[r, c] != 0:
                    continue
                cand = [v for v in range(1, 10) if is_valid(grid, r, c, v)]
                if len(cand) == 1:
                    grid[r, c] = cand[0]
                    filled += 1
                    changed = True
    return filled

def _ocr_cell_multi(cell_bgr, ocr) -> int:
    """
    OCR robuste d'une case : on génère 3 binarisations différentes et on prend
    la meilleure (ou un vote si égalité). Retourne 0 si incertain.
    """
    import cv2, numpy as np
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    if is_blank_cell(gray):
        return 0
    thrs = [
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY_INV, 11, 2),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 21, 3),
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    ]
    preds, confs = [], []
    for t in thrs:
        # mêmes critères de case vide que extract_digit
        digit28 = digit_from_binary(t)
        if digit28 is None:
            preds.append(0); confs.append(0.0); continue

        # prédiction
        try:
            prob = getattr(ocr, "predict_prob", None)
            if callable(prob):
                p = prob(digit28)  # optionnel : si tu ajoutes cette méthode au CNN
                cls = int(np.argmax(p)); conf = float(np.max(p))
            else:
                cls = int(ocr.predict_digit(digit28)); conf = 0.6 if cls != 0 else 0.0
        except Exception:
            cls, conf = 0, 0.0
        if cls == 0:
            conf = 0.0
        preds.append(cls); confs.append(conf)

    # choix : meilleur score ou vote majoritaire
    from collections import Counter
    cnt = Counter([p for p in preds if p != 0])
    if cnt:
        best_vote, votes = cnt.most_common(1)[0]
        # si majorité claire (>=2/3), prends le vote, sinon prends meilleur conf
        if votes >= 2:
            return best_vote
    # meilleur conf
    best_i = int(np.argmax(confs))
    return int(preds[best_i]) if confs[best_i] >= 0.55 else 0

def _make_ocr_backend(name: str, cfg: Dict):
    name = (name or "cnn").lower()
    if name == "cnn":
        from .ocr.cnn import CNNOCR
        mnist_weights = DEFAULTS["ocr"]["cnn_weights"]
        weights = cfg.get("ocr", {}).get("cnn_weights", mnist_weights)
        conf = cfg.get("predict", {}).get("conf_min", 0.6)
        # entraînement MNIST automatique uniquement pour le modèle MNIST : un modèle
        # custom manquant doit lever une erreur, pas être remplacé par MNIST
        return CNNOCR(weights_path=weights, train_if_missing=(weights == mnist_weights),
                      conf_min=conf)
    elif name == "tesseract":
        from .ocr.tesseract import TesseractOCR
        return TesseractOCR(psm=10, whitelist="123456789")
    else:
        raise ValueError(f"OCR backend inconnu: {name}")


def _to_grid_and_mask(cells_bgr: list[np.ndarray], ocr) -> Tuple[np.ndarray, np.ndarray]:
    """OCR sur 81 cellules → retourne (grid 9x9, given_mask 9x9 bool)."""
    grid = np.zeros((9, 9), dtype=int)
    given = np.zeros((9, 9), dtype=bool)
    for idx, cell in enumerate(cells_bgr):
        r, c = divmod(idx, 9)
        dimg = extract_digit(cell)
        if dimg is None:
            continue
        val = int(ocr.predict_digit(dimg))
        if 1 <= val <= 9:
            grid[r, c] = val
            given[r, c] = True
    return grid, given


def run(image_path: str, out_path: str, cfg: Dict):
    """
    Pipeline de bout en bout :
      1) Détection de la grille et rectification (450x450)
      2) Découpe en 81 cases + extraction chiffre
      3) OCR via backend (cnn/tesseract)
      4) Résolution Sudoku (backtracking)
      5) Réincrustation de la solution sur l'image originale
    """
    cfg = load_config(overrides=cfg)  # complète avec les défauts et valide
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    # Chargement image
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Impossible de lire l'image: {image_path}")
    print(f" Image chargée: {image_path} ({img.shape[1]}x{img.shape[0]})")

    # 1) Détection
    quad = find_sudoku_quad(img)
    if quad is None:
        raise RuntimeError("Grille Sudoku introuvable dans l'image.")
    print(" Grille détectée.")

    # 2) Rectification
    warp_size = int(cfg.get("detect", {}).get("warp_size", 450))
    warped, M, Minv = four_point_transform(img, quad, size=warp_size)

    # lignes & découpe précise
    xs, ys = detect_grid_lines(warped)
    cells = split_cells_by_lines(warped, xs, ys)

    # OCR + given mask basé sur présence d’encre, pas sur l’OCR
    grid = np.zeros((9, 9), dtype=int)
    given = np.zeros((9, 9), dtype=bool)
    backend = cfg.get("ocr", {}).get("backend", "cnn")
    ocr = _make_ocr_backend(backend, cfg)

    grid = np.zeros((9, 9), dtype=int)

    # IMPORTANT : given_mask basé sur présence d'encre, pas sur l'OCR
    given = np.zeros((9, 9), dtype=bool)

    from .cells import extract_digit  # si ce n'est pas déjà importé en haut

    for idx, cell in enumerate(cells):
        r, c = divmod(idx, 9)
        dimg = extract_digit(cell)
        if dimg is None:
            # essaye multi-OCR direct quand c'est limite
            v = _ocr_cell_multi(cell, ocr)
            if 1 <= v <= 9:
                grid[r, c] = v
                given[r, c] = True   # il y a de l'encre si on a reconnu qqchose
            continue
        given[r, c] = True
        v = int(ocr.predict_digit(dimg))
        if 1 <= v <= 9:
            grid[r, c] = v

    print("OCR effectué avec backend '%s'." % backend)
    print("Grille reconnue (avant nettoyage):")
    print(grid)

    removed = _sanitize_grid(grid)
    if removed:
        print(f"Nettoyage OCR: {removed} valeurs conflictuelles supprimées.")
    print("Grille après nettoyage:")
    print(grid)

    sing = _fill_singles(grid)
    if sing:
        print(f"Singles remplis: {sing}")

    solved = grid.copy()
    ok = solve(solved)
    if not ok:
        try:
            from .ocr.tesseract import TesseractOCR
            tesser = TesseractOCR(psm=10, whitelist="123456789")
            added = 0
            for idx, cell in enumerate(cells):
                r, c = divmod(idx, 9)
                if solved[r, c] != 0:
                    continue
                v = _ocr_cell_multi(cell, tesser)  # multi-seuils aussi pour Tesseract
                if 1 <= v <= 9 and is_valid(solved, r, c, v):
                    solved[r, c] = v
                    added += 1
            if added:
                print(f"Fallback Tesseract: +{added} cases ajoutées.")
                _fill_singles(solved)
                ok = solve(solved)
        except Exception as e:
            print(f"[warn] fallback Tesseract indisponible: {e}")

    if not ok:
        raise RuntimeError("Le sudoku n'a pas pu être résolu.")

    ov = cfg["overlay"]
    result = overlay_solution(img, Minv, solved, given,
                              color=tuple(int(v) for v in ov["color"]),
                              given_color=tuple(int(v) for v in ov["given_color"]),
                              scale=float(ov["scale"]), thickness=int(ov["thickness"]),
                              warp_size=warp_size, xs=xs, ys=ys, show_mode=ov["show_mode"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ok = cv2.imwrite(out_path, result)
    if not ok:
        raise RuntimeError(f"Échec d'écriture du fichier de sortie: {out_path}")
    print(f" Résultat écrit: {out_path}")

    print("Sudoku résolu.")
    print(f" → Entrée : {image_path}")
    print(f" → Sortie : {out_path}")
    print(f" → Backend OCR : {backend}")
    return {
        "in": image_path,
        "out": out_path,
        "backend": backend,
        "given": int(np.sum(given)),
    }
