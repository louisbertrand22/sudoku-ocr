#!/usr/bin/env python3
# scripts/train_cnn.py
"""Entraîne le CNN de lecture des chiffres (models/sudoku_cnn.keras + .meta.json).

Recette (voir docs/reseau_de_neurones.md et scripts/ocr_experiments.py) :
- chaque image de data/assets est traitée comme une case de grille et passe par
  le prétraitement réel de l'app (extract_digit -> to_28x28_white_on_black) :
  le modèle apprend sur ce qu'il verra en production ;
- 5 variantes abîmées par image (géométrie, épaisseur, flou, contraste, bruit, JPEG) ;
- séparation stratifiée : 15 % de test jamais vus (graine 42), puis 15 % du reste
  pour l'arrêt anticipé ; les métriques de test sont affichées en fin d'entraînement.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sudoku_ocr.cells import extract_digit  # noqa: E402
from sudoku_ocr.ocr.base import to_28x28_white_on_black  # noqa: E402
from sudoku_ocr.ocr.cnn import save_meta  # noqa: E402

IMG = 28
CELL = 56        # une image du dataset est agrandie en case de grille (~50 px en réalité)
CONF_MIN = 0.6   # predict.conf_min de configs/default.yaml
# momentum < 0.99 (défaut Keras) : peu de pas par époque, les statistiques
# glissantes de BatchNorm ne convergeraient pas et le modèle s'effondrerait
# en inférence (val_accuracy ~ hasard)
BN_MOMENTUM = 0.9


# ---------------------------------------------------------------- données -- #

def stratified_split(paths, labels, test_size=0.15, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_tr, idx_va = next(sss.split(paths, labels))
    return paths[idx_tr], labels[idx_tr], paths[idx_va], labels[idx_va]


def list_files_and_labels(root):
    """Lit un dossier plat contenant 1_*.jpg, 2_*.jpg, ... et retourne (paths, labels)."""
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


def load_split(root="data/assets"):
    """-> classes, (fit, y), (arrêt anticipé, y), (test, y) ; images en niveaux de gris 28x28.

    Le test est la validation historique (graine 42, 15 %) : il n'a jamais servi à
    apprendre de poids. y = indices 0..K-1 ; classes[i] = chiffre.
    """
    paths, labels = list_files_and_labels(root)
    classes = sorted(np.unique(labels).tolist())
    y = np.array([classes.index(int(l)) for l in labels], dtype=np.int64)
    tr_p, tr_y, te_p, te_y = stratified_split(paths, y, test_size=0.15, seed=42)
    fit_p, fit_y, es_p, es_y = stratified_split(tr_p, tr_y, test_size=0.15, seed=0)
    read = lambda ps: np.stack([cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in ps])
    return classes, (read(fit_p), fit_y), (read(es_p), es_y), (read(te_p), te_y)


# ---------------------------------------------------------- prétraitement -- #

def as_cell(gray28: np.ndarray) -> np.ndarray:
    return cv2.resize(gray28, (CELL, CELL), interpolation=cv2.INTER_CUBIC)


def model_input(digit28: np.ndarray) -> np.ndarray:
    """Même entrée que CNNOCR._prepare (input_range=unit, white_on_black)."""
    return to_28x28_white_on_black(digit28).astype(np.float32) / 255.0


def pipeline_style(cell_gray: np.ndarray):
    """Case (encre foncée sur fond clair) -> entrée du modèle comme dans l'app, ou None si rejetée."""
    d = extract_digit(cv2.cvtColor(cell_gray, cv2.COLOR_GRAY2BGR))
    return None if d is None else model_input(d)


def degrade(cell: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Abîme une case comme une photo ou une capture : géométrie, épaisseur, flou, bruit, JPEG."""
    h, w = cell.shape
    m = cv2.getRotationMatrix2D((w / 2 + rng.uniform(-3, 3), h / 2 + rng.uniform(-3, 3)),
                                rng.uniform(-8, 8), rng.uniform(0.85, 1.15))
    out = cv2.warpAffine(cell, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    k = rng.integers(0, 3)
    if k == 1:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))   # encre foncée : trait plus épais
    elif k == 2:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8))  # trait plus fin
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.3, 1.2))
    lo, hi = rng.uniform(0, 50), rng.uniform(185, 255)       # contraste réduit / fond grisé
    out = (lo + out.astype(np.float32) * (hi - lo) / 255.0)
    out = np.clip(out + rng.normal(0, rng.uniform(0, 6), out.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 95))])
        out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out


def to_pipeline_set(cells: np.ndarray, ys: np.ndarray):
    """Cases -> (X, y) prêts pour le modèle ; les cases rejetées par l'extraction sont écartées."""
    xs, keep = [], []
    for i, c in enumerate(cells):
        x = pipeline_style(c)
        if x is not None:
            xs.append(x); keep.append(i)
    return np.stack(xs)[..., None], ys[keep]


def augmented_set(grays: np.ndarray, ys: np.ndarray, variants: int, seed: int, extra=()):
    """Chaque image en case + `variants` versions abîmées, puis prétraitement de l'app.

    `extra` : (cases, y) supplémentaires déjà au format case (ex. chiffres synthétiques).
    """
    rng = np.random.default_rng(seed)
    cells, labels = [], []
    for g, y in zip(grays, ys):
        base = as_cell(g)
        cells.append(base); labels.append(y)
        for _ in range(variants):
            cells.append(degrade(base, rng)); labels.append(y)
    for cell, y in extra:
        cells.append(cell); labels.append(y)
    return to_pipeline_set(np.stack(cells), np.array(labels))


# ------------------------------------------------------------------ modèle -- #

def build_model(num_classes: int, bias_prior=None):
    inp = layers.Input((IMG, IMG, 1))
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x); x = layers.ReLU()(x)
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


def train(x, y, x_es, y_es, *, n_classes, batch=64, epochs=60, patience=10, seed=1,
          augment_layer=None, verbose=0):
    """Entraîne avec arrêt anticipé sur (x_es, y_es) ; renvoie (modèle, nb d'époques)."""
    keras.utils.set_random_seed(seed)
    model = build_model(n_classes)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=seed)
    if augment_layer is not None:
        ds = ds.map(lambda a, b: (augment_layer(a, training=True), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    cbs = [keras.callbacks.ReduceLROnPlateau(patience=max(3, patience // 2), factor=0.5, min_lr=1e-5),
           keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)]
    hist = model.fit(ds, validation_data=(x_es, y_es), epochs=epochs, callbacks=cbs, verbose=verbose)
    return model, len(hist.history["loss"])


def evaluate(model, xs, ys) -> dict:
    """xs : liste d'entrées 28x28 (None = rejet par l'extraction, compté comme erreur)."""
    idx = [i for i, x in enumerate(xs) if x is not None]
    pred = np.full(len(ys), -1)
    conf = np.zeros(len(ys))
    if idx:
        p = model.predict(np.stack([xs[i] for i in idx])[..., None], verbose=0, batch_size=512)
        pred[idx], conf[idx] = p.argmax(1), p.max(1)
    ok = conf >= CONF_MIN
    return {
        "acc": float((pred == ys).mean()),                       # toutes réponses (argmax)
        "couverture": float(ok.mean()),                          # réponses acceptées (conf >= 0.6)
        "acc_acceptees": float((pred[ok] == ys[ok]).mean()) if ok.any() else 0.0,
        "erreurs_acceptees": int(((pred != ys) & ok).sum()),     # chiffres faux inscrits
        "conf_mediane": float(np.median(conf[idx])) if idx else 0.0,
        "rejets_extraction": len(ys) - len(idx),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/assets", help="Dossier avec 1_*.jpg, 2_*.jpg, ...")
    ap.add_argument("--out", default="models/sudoku_cnn.keras")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10, help="arrêt anticipé")
    ap.add_argument("--variants", type=int, default=5, help="versions abîmées par image")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    classes, (g_fit, y_fit), (g_es, y_es), (g_te, y_te) = load_split(args.data)
    print(f"Classes : {classes} | entraînement {len(y_fit)} | arrêt anticipé {len(y_es)} | test {len(y_te)}")
    x_fit, y_fit = augmented_set(g_fit, y_fit, args.variants, args.seed)
    x_es, y_es = to_pipeline_set(np.stack([as_cell(g) for g in g_es]), y_es)
    print(f"Images d'entraînement après augmentation : {len(y_fit)}")

    model, n_ep = train(x_fit, y_fit, x_es, y_es, n_classes=len(classes), batch=args.batch,
                        epochs=args.epochs, patience=args.patience, seed=args.seed, verbose=2)

    m = evaluate(model, [pipeline_style(as_cell(g)) for g in g_te], y_te)
    print(f"\nTest ({len(y_te)} images jamais vues, prétraitement de l'app) après {n_ep} époques :")
    print(f"  précision {m['acc']:.4f} | couverture (conf >= {CONF_MIN}) {m['couverture']:.4f} | "
          f"précision des acceptées {m['acc_acceptees']:.4f} | chiffres faux acceptés "
          f"{m['erreurs_acceptees']} | rejets extraction {m['rejets_extraction']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    # contrat d'inférence lu par CNNOCR : index de sortie -> chiffre, échelle, polarité
    meta_path = save_meta(args.out, {
        "classes": [int(c) for c in classes],
        "input_range": "unit",
        "polarity": "white_on_black",
    })
    print(f"✅ Modèle sauvegardé: {args.out} (+ {meta_path})")


if __name__ == "__main__":
    main()
