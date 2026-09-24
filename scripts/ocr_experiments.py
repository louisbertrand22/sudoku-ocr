#!/usr/bin/env python3
"""Banc d'essai du CNN : compare des recettes d'entraînement sur des mesures communes.

Protocole
- test : la validation historique de train_cnn.py (split stratifié, graine 42, 15 %).
  Le modèle actuel ne l'a jamais vu en apprentissage (seulement pour l'arrêt anticipé).
- les recettes s'entraînent sur les 85 % restants, dont 15 % servent à leur propre
  arrêt anticipé : elles ne voient jamais le test.
- trois mesures par modèle :
    test_brut      prétraitement de l'entraînement historique (niveaux de gris inversés)
    test_pipeline  prétraitement réel de l'app (extract_digit -> to_28x28_white_on_black) ;
                   une case rejetée par l'extraction compte comme une erreur
    grilles        les cases chiffrées des grilles de data/samples (lues comme dans l'app)

Usage : python scripts/ocr_experiments.py [--seeds 0 1] [--only E2 E3] [--out data/outputs/exp]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

from sudoku_ocr.cells import extract_digit  # noqa: E402
from sudoku_ocr.pipeline import read_grid  # noqa: E402
from train_cnn import (CELL, as_cell, augmented_set, degrade, evaluate, load_split,  # noqa: E402
                       model_input, pipeline_style, to_pipeline_set, train)
EXPECTED = {
    "sudoku2": "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
    "sudoku3": "543070619102069703607000004250080970074205800801006540708641300305928467400037120",
    "sudoku4": "205308409070000050904000607500040002000507000600030008406000801020000060801209704",
}


# ------------------------------------------------------------ prétraitements -- #

def raw_style(gray28: np.ndarray) -> np.ndarray:
    """Prétraitement historique de train_cnn.py : pixels 0..1, chiffre blanc sur noir."""
    return 1.0 - gray28.astype(np.float32) / 255.0


# -------------------------------------------------------------- augmentation -- #

def list_fonts(limit: int, seed: int = 0) -> list[str]:
    """Polices système droites capables d'afficher les chiffres, une par famille."""
    from PIL import Image, ImageDraw, ImageFont
    out = subprocess.run(["fc-list", ":", "family", "style", "file"], capture_output=True, text=True).stdout
    skip = ("emoji", "symbol", "icon", "awesome", "math", "dingbat", "braille", "cjk", "propo",
            "italic", "oblique")
    latin_noto = {"Noto Sans", "Noto Serif", "Noto Sans Mono", "Noto Sans Display", "Noto Serif Display"}

    def render(font, ch):
        img = Image.new("L", (48, 48), 0)
        ImageDraw.Draw(img).text((4, 0), ch, fill=255, font=font)
        return np.array(img)

    by_family: dict[str, str] = {}
    for line in out.splitlines():
        path, _, rest = line.partition(":")
        fam = rest.split(":")[0].strip().split(",")[0]
        if (not path.lower().endswith((".ttf", ".otf")) or any(k in line.lower() for k in skip)
                or (fam.startswith("Noto") and fam not in latin_noto)):
            continue
        by_family.setdefault(fam, path.strip())
    fonts = []
    for path in sorted(by_family.values()):
        try:
            f = ImageFont.truetype(path, 40)
            glyphs = [render(f, str(d)) for d in range(1, 10)]
            tofu = render(f, "\ue000")  # caractère absent : sert à repérer les glyphes manquants
            if all(g.sum() > 0 and not np.array_equal(g, tofu) for g in glyphs) and \
                    len({g.tobytes() for g in glyphs}) == 9:
                fonts.append(path)
        except Exception:
            continue
    random.Random(seed).shuffle(fonts)
    return fonts[:limit]


def render_digit(digit: int, font_path: str, rng: np.random.Generator) -> np.ndarray:
    """Chiffre imprimé synthétique dans une case CELL x CELL (encre foncée sur fond clair)."""
    from PIL import Image, ImageDraw, ImageFont
    bg = int(rng.integers(200, 256))
    img = Image.new("L", (CELL, CELL), bg)
    size = int(CELL * rng.uniform(0.45, 0.75))
    font = ImageFont.truetype(font_path, size)
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = draw.textbbox((0, 0), str(digit), font=font)
    x = (CELL - (x1 - x0)) / 2 - x0 + rng.uniform(-3, 3)
    y = (CELL - (y1 - y0)) / 2 - y0 + rng.uniform(-3, 3)
    draw.text((x, y), str(digit), fill=int(rng.integers(0, 50)), font=font)
    return np.array(img)


# ---------------------------------------------------------------- données -- #

def real_grid_cells(classes):
    """Cases chiffrées des grilles d'exemple, lues comme dans l'app ; None = case manquée."""
    class NoOCR:
        def predict_digit(self, _):
            return 0
    xs, ys = [], []
    for name, s in EXPECTED.items():
        truth = [int(c) for c in s]
        for suffix in ("", "_dark"):
            img = cv2.imread(f"data/samples/{name}{suffix}.png")
            reading = read_grid(img, None, ocr=NoOCR())
            for cell, t in zip(reading.cells, truth):
                if t:
                    d = extract_digit(cell)
                    xs.append(None if d is None else model_input(d))
                    ys.append(classes.index(t))
    return xs, np.array(ys)


# ------------------------------------------------------------- recettes -- #

KERAS_AUG = keras.Sequential([
    layers.RandomRotation(0.05), layers.RandomTranslation(0.05, 0.05),
    layers.RandomZoom(0.10), layers.RandomContrast(0.20),
])


def build_sets(recipe, fit_set, es_set, classes, seed, fonts):
    """Données d'entraînement et d'arrêt anticipé selon la recette."""
    (g_fit, y_fit), (g_es, y_es) = fit_set, es_set
    if recipe["style"] == "raw":
        return (raw_style(g_fit)[..., None], y_fit, raw_style(g_es)[..., None], y_es)
    rng = np.random.default_rng(seed + 1000)  # flux distinct de celui des variantes
    synth = [(degrade(render_digit(digit, fonts[int(rng.integers(len(fonts)))], rng), rng), ci)
             for _ in range(recipe.get("synth_per_digit", 0)) for ci, digit in enumerate(classes)]
    x_fit, y_fit2 = augmented_set(g_fit, y_fit, recipe["variants"], seed, extra=synth)
    x_es, y_es2 = to_pipeline_set(np.stack([as_cell(g) for g in g_es]), y_es)
    return x_fit, y_fit2, x_es, y_es2


RECIPES = {
    "E1_actuelle": dict(style="raw", batch=256, epochs=20, patience=6, keras_aug=True),
    "E2_plus_longue": dict(style="raw", batch=64, epochs=60, patience=10, keras_aug=True),
    # E3 = recette de train_cnn.py (retenue)
    "E3_pretraitement_aligne": dict(style="pipeline", variants=5, batch=64, epochs=60, patience=10,
                                    keras_aug=False),
    "E4_aligne_synthetique": dict(style="pipeline", variants=5, synth_per_digit=600, batch=64,
                                  epochs=60, patience=10, keras_aug=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--out", default="data/outputs/exp")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    classes, fit_set, es_set, (g_te, y_te) = load_split("data/assets")
    tests = {
        "test_brut": ([raw_style(g) for g in g_te], y_te),
        "test_pipeline": ([pipeline_style(as_cell(g)) for g in g_te], y_te),
        "grilles": real_grid_cells(classes),
    }
    print(f"entraînement {len(fit_set[1])} | arrêt anticipé {len(es_set[1])} | test {len(y_te)} | "
          f"cases réelles {len(tests['grilles'][1])} | rejets extraction test "
          f"{sum(x is None for x in tests['test_pipeline'][0])}", flush=True)
    fonts = list_fonts(80)
    print(f"polices pour le synthétique : {len(fonts)}", flush=True)

    results = []
    if not args.only or "E0_modele_actuel" in args.only:
        model = keras.models.load_model("models/sudoku_cnn.keras", compile=False)
        row = {"recette": "E0_modele_actuel", "graine": "-", "epoques": "-", "n_train": 3533, "duree_s": 0}
        row.update({k: evaluate(model, *v) for k, v in tests.items()})
        results.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)

    for name, recipe in RECIPES.items():
        if args.only and name not in args.only:
            continue
        for seed in args.seeds:
            t0 = time.time()
            x_fit, y_fit, x_es, y_es = build_sets(recipe, fit_set, es_set, classes, seed, fonts)
            model, n_ep = train(x_fit, y_fit, x_es, y_es, n_classes=len(classes), batch=recipe["batch"],
                                epochs=recipe["epochs"], patience=recipe["patience"], seed=seed,
                                augment_layer=KERAS_AUG if recipe["keras_aug"] else None)
            row = {"recette": name, "graine": seed, "epoques": n_ep, "n_train": len(y_fit),
                   "duree_s": round(time.time() - t0)}
            row.update({k: evaluate(model, *v) for k, v in tests.items()})
            model.save(out / f"{name}_s{seed}.keras")
            results.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n| recette | graine | époques | test brut | test pipeline | couverture | erreurs acceptées | cases réelles |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        tp, g = r["test_pipeline"], r["grilles"]
        print(f"| {r['recette']} | {r['graine']} | {r['epoques']} | {r['test_brut']['acc']:.3f} | "
              f"{tp['acc']:.3f} | {tp['couverture']:.3f} | {tp['erreurs_acceptees']} | "
              f"{g['acc']:.3f} ({g['erreurs_acceptees']} err.) |")


if __name__ == "__main__":
    main()
