#!/usr/bin/env python3
import os, glob, random, cv2, numpy as np
from tensorflow import keras
from tensorflow.keras import layers

IMG = 28

def load_pairs(root, n=128, seed=0):
    files = sorted(glob.glob(os.path.join(root, "*.jpg")) + glob.glob(os.path.join(root, "*.png")))
    if not files: raise SystemExit(f"Aucune image dans {root}")
    random.Random(seed).shuffle(files)
    files = files[:n]
    X, y = [], []
    for p in files:
        base = os.path.basename(p)
        try:
            lab = int(base.split("_", 1)[0])
        except:
            continue
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        im = cv2.resize(im, (IMG, IMG), interpolation=cv2.INTER_AREA)
        im = im.astype(np.float32) / 255.0
        # Inversion heuristique: si fond clair très dominant, on inverse
        if im.mean() > 0.6:
            im = 1.0 - im
        X.append(im[..., None])
        y.append(lab)
    X = np.stack(X, 0)
    y = np.array(y, dtype=np.int64)
    # Remappe  labels (ex: 1..9) -> indices 0..K-1
    classes = sorted(np.unique(y).tolist())
    idx_map = {c:i for i,c in enumerate(classes)}
    y = np.array([idx_map[int(v)] for v in y], dtype=np.int64)
    return X, y, len(classes)

def build_model(num_classes):
    inp = layers.Input((IMG, IMG, 1))
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])
    return m

def main():
    X, y, K = load_pairs("data/assets", n=128)
    # split 90/10 mais on va entraîner et valider sur le même pour sur-apprendre
    m = build_model(K)
    m.fit(X, y, epochs=100, batch_size=32, verbose=0, validation_data=(X, y))
    loss, acc = m.evaluate(X, y, verbose=0)
    print("Overfit numpy acc:", acc)

if __name__ == "__main__":
    main()
