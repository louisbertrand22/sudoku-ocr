# base OCR class
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import cv2

class OCRBase(ABC):
    """
    Interface de base pour un backend OCR de chiffres Sudoku.
    Implémentations attendues : CNN (MNIST) ou Tesseract.
    
    Contrat :
      - input : image 28x28 (np.uint8), chiffre blanc sur fond noir
      - output : int in {0..9} (0 = vide/incertain)
    """

    @abstractmethod
    def predict_digit(self, img28: np.ndarray) -> int:
        """Prédit un chiffre unique (0 si vide/incertain)."""
        raise NotImplementedError

    def batch_predict(self, imgs28: list[np.ndarray]) -> list[int]:
        """Prédiction par lot, par défaut boucle sur predict_digit."""
        return [self.predict_digit(im) for im in imgs28]


# ------------------------- utilitaires d'images ------------------------- #

def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Force un tableau numpy en niveaux de gris (uint8)."""
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("image: dimensions non supportées")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def to_28x28_white_on_black(img: np.ndarray) -> np.ndarray:
    """
    Convertit une image quelconque de digit en 28x28 (uint8),
    chiffre blanc sur fond noir. N'applique pas de binarisation agressive.
    """
    gray = ensure_gray(img)
    # Normalise la taille (garde le ratio) puis centre sur un canvas 28x28
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("image vide")
    scale = 20.0 / max(h, w)  # laisse une marge
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - nh) // 2
    x_off = (28 - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = small

    # S'assure que le chiffre est clair sur fond sombre (si l'histogramme l'indique)
    mean = float(canvas.mean())
    if mean > 127:  # probablement fond clair
        canvas = 255 - canvas

    return canvas


def postprocess_digit(pred: int) -> int:
    """Clamp de sécurité pour renvoyer 0..9."""
    if not isinstance(pred, (int, np.integer)):
        return 0
    return int(pred) if 0 <= int(pred) <= 9 else 0
