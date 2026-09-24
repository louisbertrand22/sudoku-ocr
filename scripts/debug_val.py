# scripts/debug_val.py
# Métriques d'un modèle sur le jeu de test (624 images jamais vues), avec le prétraitement de l'app.
# Usage : python scripts/debug_val.py [models/sudoku_cnn.keras]
import sys

import numpy as np
from tensorflow import keras
from train_cnn import CONF_MIN, as_cell, evaluate, load_split, pipeline_style

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/sudoku_cnn.keras"
model = keras.models.load_model(model_path, compile=False)
classes, _, _, (g_te, y_te) = load_split("data/assets")
xs = [pipeline_style(as_cell(g)) for g in g_te]

m = evaluate(model, xs, y_te)
print(f"\nTest : {len(y_te)} images ({m['rejets_extraction']} rejetées par l'extraction, comptées fausses)")
print(f"Précision           : {m['acc']:.4f}")
print(f"Confiance médiane   : {m['conf_mediane']:.3f}")
print(f"Couverture (>= {CONF_MIN}) : {m['couverture']:.4f}")
print(f"Précision acceptées : {m['acc_acceptees']:.4f} | chiffres faux acceptés : {m['erreurs_acceptees']}")

idx = [i for i, x in enumerate(xs) if x is not None]
pred = model.predict(np.stack([xs[i] for i in idx])[..., None], verbose=0).argmax(1)
y = y_te[idx]
try:
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n" + classification_report(y, pred, target_names=[str(c) for c in classes], digits=3))
    print("Matrice de confusion (lignes = vrai, colonnes = prédit), chiffres", classes)
    print(confusion_matrix(y, pred))
except ImportError:
    print("(installez scikit-learn pour le rapport par chiffre et la matrice de confusion)")
