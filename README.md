# sudoku-ocr

Détecte une grille de sudoku dans une image, lit les chiffres (CNN ou Tesseract),
résout la grille et réécrit la solution sur l'image d'origine.

## Installation

TensorFlow n'est pas publié pour Python 3.14 : utiliser Python 3.10 à 3.13.

```bash
make venv install          # uv venv -p 3.12 .venv + pip install -e ".[tesseract,train,ui,dev]"
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

La configuration est construite par couches, chacune ne remplaçant que les clés
qu'elle contient : valeurs par défaut du code, puis `configs/default.yaml` (toujours
lu s'il existe dans le dossier courant), puis le fichier `--config`, puis les options
`--weights` / `--backend`. Un `ma_config.yaml` peut donc ne contenir que ce qui change.
Les clés acceptées sont dans `src/sudoku_ocr/config.py` ; une clé inconnue est refusée.

Debug : `SUDOKU_DEBUG=1` écrit les images intermédiaires de la détection dans
`data/outputs/debug/` (ou `SUDOKU_DEBUG_DIR`).

## Interface web

```bash
make ui                    # ou : sudoku-ocr-ui  -> http://localhost:8501
```

Choisir une image d'exemple ou importer une photo, puis :
1. vérifier la grille lue : la vue « Lecture » montre la grille redressée, les lignes
   détectées et les chiffres retenus ; « Détection » montre le contour trouvé ;
2. corriger au besoin une case dans le tableau (vider une case = case vide) ; les
   conflits sont signalés et surlignés, ainsi qu'une grille à plusieurs solutions
   (souvent un chiffre manqué) ;
3. la solution s'affiche dès que la grille est valide et à solution unique, et se
   télécharge en PNG.

La barre latérale reprend `configs/default.yaml` (modèle, confiance, affichage).

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
src/sudoku_ocr/            le paquet (ui/ : interface Streamlit)
tests/
```
