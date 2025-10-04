# scripts/dump_preprocessed.py
import os, tensorflow as tf
from train_cnn import make_ds

os.makedirs("data/outputs/debug_pp", exist_ok=True)

# MÊMES options que ton entraînement actuel !
ds_tr, ds_va, _, _ = make_ds(
    root="data/assets",
    batch=32,
    val_split=0.15,
    seed=42,
    invert_mode="none",   # mets "fixed" ou "auto" si c'est ce que tu utilises
    use_aug=False         # val sans augmentation
)

x_val, y_val = next(iter(ds_va))
# remet en 0..255 uint8 pour sauver des PNG
x = tf.clip_by_value(x_val * 255.0, 0, 255)
x = tf.cast(x, tf.uint8).numpy()

import cv2
for i in range(min(16, x.shape[0])):
    cv2.imwrite(f"data/outputs/debug_pp/val_{i}_label_{int(y_val[i].numpy())}.png", x[i])
print("Saved -> data/outputs/debug_pp/")
