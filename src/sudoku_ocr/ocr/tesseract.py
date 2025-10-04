# Tesseract OCR wrapper
from __future__ import annotations
import shutil
import numpy as np
import cv2


try:
    import pytesseract # type: ignore
    TESS_AVAILABLE = True
except Exception: # pragma: no cover
    TESS_AVAILABLE = False


from .base import OCRBase, to_28x28_white_on_black, postprocess_digit




def _is_tesseract_installed() -> bool:
    """Vérifie la présence du binaire tesseract dans le PATH."""
    return shutil.which("tesseract") is not None and TESS_AVAILABLE

class TesseractOCR(OCRBase):
    """
    Backend OCR via Tesseract (pytesseract).


    Conseils:
    - psm=10 (single char) pour un chiffre isolé
    - whitelist restreinte à 1..9 (pas de 0 en Sudoku)
    """


    def __init__(self, psm: int = 10, whitelist: str = "123456789"):
        if not _is_tesseract_installed():
            raise RuntimeError(
            "Tesseract n'est pas disponible. Installez le binaire 'tesseract' et la lib pytesseract."
            )
        self.psm = int(psm)
        self.whitelist = whitelist


    def predict_digit(self, img28: np.ndarray) -> int:
        if img28 is None:
            return 0
        # Préprocess : agrandir légèrement pour aider Tesseract
        x = to_28x28_white_on_black(img28)
        x = cv2.copyMakeBorder(x, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        x = cv2.resize(x, (56, 56), interpolation=cv2.INTER_NEAREST)


        # Binarisation douce
        _, bw = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        config = f"--psm {self.psm} -c tessedit_char_whitelist={self.whitelist}"
        txt = pytesseract.image_to_string(bw, config=config)
        # Garder un seul chiffre valide si présent
        digits = [ch for ch in txt if ch in self.whitelist]
        if len(digits) == 1:
            return postprocess_digit(int(digits[0]))
        return 0