#!/usr/bin/env python3
# scripts/train_from_folder.py
import os, glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedShuffleSplit

IMG = 28

def stratified_split(paths, labels, test_size=0.15, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_tr, idx_va = next(sss.split(paths, labels))
    return paths[idx_tr], labels[idx_tr], paths[idx_va], labels[idx_va]

def list_files_and_labels(root):
    """Lit un dossier plat contenant 0_*.jpg, 1_*.jpg, ... et retourne (paths, labels)."""
    files = sorted(glob.glob(os.path.join(root, "*.jpg")) + glob.glob(os.path.join(root, "*.png")))
    paths, labels = [], []
    for p in files:
        base = os.path.basename(p)
        try:
            lab = int(base.split("_", 1)[0])
        except Exception:
            continue
        paths.append(p); labels.append(lab)
    if not paths:
        raise SystemExit(f"Aucune image trouvée dans {root}")
    return np.array(paths), np.array(labels, dtype=np.int32)

def make_ds(root: str,
            batch: int = 256,
            val_split: float = 0.15,
            seed: int = 42,
            invert_mode: str = "fixed",  # "none" | "fixed" | "auto"
            use_aug: bool = True,
            drop_zero: bool = False):
    paths, labels = list_files_and_labels(root)

    # Option: retirer la classe 0 du dataset (on gérera le "blank" autrement)
    if drop_zero:
        keep = labels != 0
        paths, labels = paths[keep], labels[keep]

    uniq, cnt = np.unique(labels, return_counts=True)
    print("Classes trouvées:", uniq.tolist())
    print("Distribution   :", {int(u): int(c) for u, c in zip(uniq, cnt)})

    # mapping compact 0..K-1
    sorted_classes = sorted(uniq.tolist())
    cls_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    idx_to_cls = {i: c for c, i in cls_to_idx.items()}
    y_all = np.array([cls_to_idx[int(l)] for l in labels], dtype=np.int32)
    num_classes = len(sorted_classes)
    print("Mapping classes -> indices:", cls_to_idx, f"(#classes = {num_classes})")

    # split STRATIFIÉ
    tr_paths, tr_y, val_paths, val_y = stratified_split(paths, y_all, test_size=val_split, seed=seed)
    print("Train dist:", dict(zip(*np.unique(tr_y, return_counts=True))))
    print("Val   dist:", dict(zip(*np.unique(val_y, return_counts=True))))

    def decode_img(path):
        img_bytes = tf.io.read_file(path)
        # si 100% jpg, tu peux mettre decode_jpeg; sinon garde decode_image
        img = tf.io.decode_image(img_bytes, channels=1, expand_animations=False)
        img = tf.image.resize(img, (IMG, IMG), method="area")
        img = tf.cast(img, tf.float32) / 255.0
        # Polarité
        if invert_mode == "fixed":
            img = 1.0 - img                # impose chiffre blanc/fond noir
        elif invert_mode == "auto":
            mean = tf.reduce_mean(img)
            img = tf.cond(mean > 0.6, lambda: 1.0 - img, lambda: img)
        # pas d’autres bidouilles tant que la val n’est pas stable
        return img

    aug = keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.20),
    ])

    def pipeline(P, Y, training: bool):
        ds = tf.data.Dataset.from_tensor_slices((P, Y))
        def load_one(p, y):
            x = decode_img(p)
            if training and use_aug:
                x = aug(x, training=True)
            return x, y
        if training:
            ds = ds.shuffle(4096, seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(load_one, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    ds_tr = pipeline(tr_paths, tr_y, True)
    ds_va = pipeline(val_paths, val_y, False)

    print("Exemples:", list(zip(tr_paths[:5].tolist(), tr_y[:5].tolist())))
    return ds_tr, ds_va, num_classes, tr_y, idx_to_cls


def build_model(num_classes: int, bias_prior=None):
    inp = layers.Input((IMG, IMG, 1))
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    if bias_prior is not None:
        bias_init = keras.initializers.Constant(bias_prior)
    else:
        bias_init = "zeros"

    out = layers.Dense(num_classes, activation="softmax", bias_initializer=bias_init)(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dossier avec 0_*.jpg, 1_*.jpg, ...")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="models/sudoku_cnn.keras")
    ap.add_argument("--invert", choices=["none", "fixed", "auto"], default="fixed",
                    help="Polarité des images: none | fixed (1-img) | auto (seuil sur la moyenne)")
    ap.add_argument("--no-aug", action="store_true", help="Désactive les augmentations data.")
    ap.add_argument("--class-weight", action="store_true", help="Active un équilibrage par classe.")
    ap.add_argument("--drop-zero", action="store_true",
                    help="Exclut la classe 0 (blank) du training.")
    ap.add_argument("--bias-prior", action="store_true",
                    help="Initialise le biais de la dernière couche avec les priors de classes (train).")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ds_tr, ds_va, num_classes, tr_y, cls_idx_to_label = make_ds(
        args.data, batch=args.batch,
        val_split=0.15, seed=42,
        invert_mode=args.invert, use_aug=not args.no_aug,
        drop_zero=args.drop_zero
    )
    bias_prior = None
    if args.bias_prior:
        counts = np.bincount(tr_y, minlength=num_classes).astype(np.float32)
        priors = counts / counts.sum()
        bias_prior = np.log(priors + 1e-8)  # log-odds multiclasses

    model = build_model(num_classes, bias_prior=bias_prior)

    # model = build_model(num_classes)
    cbs = [
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5),
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(args.out, save_best_only=True),
    ]

    fit_kwargs = dict(validation_data=ds_va, epochs=args.epochs, callbacks=cbs, verbose=2)
    if args.class_weight:
        uniq, cnt = np.unique(tr_y, return_counts=True)
        maxc = cnt.max()
        class_weight = {int(i): float(maxc / c) for i, c in zip(uniq, cnt)}
        print("Class weights:", class_weight)
        fit_kwargs["class_weight"] = class_weight

    model.fit(ds_tr, **fit_kwargs)
    model.save(args.out)
    print(f"✅ Modèle sauvegardé: {args.out}")

if __name__ == "__main__":
    main()
