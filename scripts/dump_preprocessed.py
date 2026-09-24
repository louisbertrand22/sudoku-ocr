# scripts/dump_preprocessed.py
# Enregistre des exemples d'entraînement tels que le réseau les voit (après augmentation et
# prétraitement de l'app), pour vérifier visuellement les données.
import os

import cv2
import numpy as np
from train_cnn import augmented_set, load_split

OUT = "data/outputs/debug_pp"
os.makedirs(OUT, exist_ok=True)

classes, (g_fit, y_fit), _, _ = load_split("data/assets")
x, y = augmented_set(g_fit[:6], y_fit[:6], variants=5, seed=1)
for i, (img, lab) in enumerate(zip(x, y)):
    cv2.imwrite(f"{OUT}/{i:02d}_chiffre_{classes[int(lab)]}.png", (img[..., 0] * 255).astype(np.uint8))
print(f"{len(x)} images -> {OUT}/")
