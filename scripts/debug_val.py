# scripts/debug_val.py
import numpy as np, tensorflow as tf
from tensorflow import keras
from train_cnn import make_ds  # importe depuis ton script

model = keras.models.load_model("models/sudoku_cnn.keras")

# garde EXACTEMENT les mêmes options que pour l'entraînement
ds_tr, ds_va, num_classes, _ = make_ds(
    root="data/assets",
    batch=256,
    val_split=0.15,
    seed=42,
    invert_mode="fixed",   # adapte si tu as changé
    use_aug=False
)

x_val, y_val = next(iter(ds_va))
probs = model.predict(x_val, verbose=0)
pred = np.argmax(probs, axis=1)

# distributions
uy, cy = np.unique(y_val.numpy(), return_counts=True)
up, cp = np.unique(pred, return_counts=True)
print("Val labels dist:", dict(zip(uy.tolist(), cy.tolist())))
print("Pred dist      :", dict(zip(up.tolist(), cp.tolist())))

# petite matrice de confusion (pratique)
try:
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_val.numpy(), pred))
except Exception:
    pass
