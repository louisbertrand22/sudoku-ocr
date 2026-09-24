# scripts/debug_val.py
# Métriques du modèle sur tout le jeu de validation (mêmes options que l'entraînement).
# Usage : python scripts/debug_val.py [models/sudoku_cnn.keras]
import sys

import numpy as np
from tensorflow import keras
from train_cnn import make_ds  # importe depuis ton script

CONF_MIN = 0.6  # predict.conf_min de configs/default.yaml

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/sudoku_cnn.keras"
model = keras.models.load_model(model_path, compile=False)

# garde EXACTEMENT les mêmes options que pour l'entraînement
_, ds_va, num_classes, _, idx_to_cls = make_ds(
    root="data/assets",
    batch=256,
    val_split=0.15,
    seed=42,
    invert_mode="fixed",   # adapte si tu as changé
    use_aug=False
)

xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in ds_va])
x_val, y_val = np.concatenate(xs), np.concatenate(ys)
probs = model.predict(x_val, verbose=0)
pred, conf = probs.argmax(axis=1), probs.max(axis=1)
digits = [idx_to_cls[i] for i in range(num_classes)]

print(f"\nValidation : {len(y_val)} images")
print(f"Précision           : {(pred == y_val).mean():.4f}")
accepted = conf >= CONF_MIN
print(f"Confiance médiane   : {np.median(conf):.3f}")
print(f"Sous le seuil {CONF_MIN}  : {100 * (~accepted).mean():.1f} % des images")
print(f"Précision acceptées : {(pred[accepted] == y_val[accepted]).mean():.4f}")

try:
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n" + classification_report(y_val, pred, target_names=[str(d) for d in digits], digits=3))
    print("Matrice de confusion (lignes = vrai, colonnes = prédit), chiffres", digits)
    print(confusion_matrix(y_val, pred))
except ImportError:
    print("(installez scikit-learn pour le rapport par chiffre et la matrice de confusion)")
