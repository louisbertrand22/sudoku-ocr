# sudoku-ocr

Détecte une grille de sudoku dans une image, lit les chiffres (CNN ou Tesseract),
résout la grille et réécrit la solution sur l'image d'origine.

## Installation

TensorFlow n'est pas publié pour Python 3.14 : utiliser Python 3.10 à 3.13.

```bash
make venv install          # uv venv -p 3.12 .venv + pip install -e ".[tesseract,train,dev]"
```

Sans `make` : `pip install -e ".[train,dev]"` dans un environnement Python ≤ 3.13.
L'extra `tesseract` demande aussi le binaire `tesseract` installé sur le système.

## Modèle

Le modèle n'est pas versionné. Pour l'entraîner sur `data/assets` (~30 s sur CPU) :

```bash
make train                 # -> models/sudoku_cnn.keras + models/sudoku_cnn.meta.json
```

Le fichier `.meta.json` décrit le contrat du modèle (chiffre associé à chaque sortie,
échelle et polarité des pixels) ; il est lu à l'inférence et doit accompagner le `.keras`.

## Utilisation

```bash
sudoku-ocr --image data/samples/sudoku4.png --out data/outputs/result.jpg
sudoku-ocr --image photo.jpg --config ma_config.yaml
sudoku-ocr --image photo.jpg --weights autre_modele.keras --backend cnn
```

La configuration est lue dans `configs/default.yaml` (ou `--config`), puis surchargée
par les options. Les clés acceptées et leurs valeurs par défaut sont dans
`src/sudoku_ocr/config.py` ; une clé inconnue est refusée.

Debug : `SUDOKU_DEBUG=1` écrit les images intermédiaires de la détection dans
`data/outputs/debug/` (ou `SUDOKU_DEBUG_DIR`).

## Tests

```bash
make test                  # tout ; les tests e2e sont ignorés si le modèle est absent
make test-fast             # sans le modèle CNN
SUDOKU_OCR_WEIGHTS=chemin/modele.keras pytest -m e2e
```

Les tests e2e vérifient la lecture exacte des chiffres de `data/samples/*.png`
et la validité des solutions.

## Fonctionnement

1. **Détection** (`detect.py`) : contours de l'image, chaque quadrilatère candidat est
   noté par sa ressemblance à une grille 9×9 une fois redressé (`cells.grid_score`).
2. **Redressement** (`geometry.py`) en carré 450×450, puis détection des lignes et
   découpe des 81 cases (`cells.py`).
3. **Cases vides** : contraste du centre de la case, puis filtrage des composantes
   (fragments de lignes, bords de zones grisées).
4. **OCR** (`ocr/`) : CNN Keras, vote sur plusieurs binarisations en cas de doute,
   repli sur Tesseract si la grille ne se résout pas.
5. **Résolution** (`solver.py`) : backtracking, case la plus contrainte d'abord.
6. **Réincrustation** (`overlay.py`) de la solution sur l'image d'origine.

## Arborescence

```
configs/default.yaml       configuration par défaut
data/assets/               chiffres 1..9 pour l'entraînement (<chiffre>_<id>.jpg)
data/assets_zero_backup/   cases vides (classe 0), écartées de l'entraînement
data/assets_trash/         images rejetées (floues / peu contrastées)
data/samples/              images de grilles d'exemple (tests e2e)
scripts/                   entraînement et outils sur le dataset
src/sudoku_ocr/            le paquet
tests/
```
