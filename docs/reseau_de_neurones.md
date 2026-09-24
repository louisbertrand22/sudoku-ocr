# Le réseau de neurones de sudoku-ocr

> Toutes les valeurs de ce document ont été mesurées sur le modèle versionné
> `models/sudoku_cnn.keras` (commit `7d7bdf6`) et sur le jeu de validation exact
> utilisé à l'entraînement. Les commandes pour les reproduire sont en fin de document.

## 1. Ce que fait (et ne fait pas) le réseau

Le réseau **ne résout pas la grille** : il **lit les chiffres**. La résolution est
faite ensuite par un algorithme classique, sans apprentissage.

```
photo ──► détection de la grille ──► 81 cases ──► RÉSEAU DE NEURONES ──► grille 9×9 ──► solveur ──► solution
          (OpenCV, detect.py)        (cells.py)    une case -> un chiffre    (chiffres lus)   (backtracking,
                                                   (ocr/cnn.py)                               solver.py)
```

Pour chaque case non vide, le réseau reçoit une petite image 28×28 du chiffre et
répond « c'est un 1, un 2, … ou un 9 », avec une probabilité pour chacun.
C'est un problème de **classification d'images en 9 classes**.

Le solveur (`solver.py`) est un **backtracking** : il choisit la case vide qui a le
moins de candidats possibles (heuristique MRV), essaie chaque candidat, et revient
en arrière en cas d'impasse. Il est exact : si la grille lue est juste, la solution
l'est aussi. Toute erreur finale vient donc de la lecture, c'est-à-dire du réseau
ou de l'extraction des cases.

## 2. Architecture

Un **réseau de neurones convolutif (CNN)** : trois blocs « convolution → normalisation →
activation », puis une couche de décision.

```
entrée 28×28×1 (niveaux de gris, chiffre blanc sur fond noir, pixels 0..1)
   │
   ├─ Conv 3×3, 32 filtres ─ BatchNorm ─ ReLU      → 28×28×32
   ├─ MaxPooling 2×2                                → 14×14×32
   ├─ Conv 3×3, 64 filtres ─ BatchNorm ─ ReLU      → 14×14×64
   ├─ MaxPooling 2×2                                →  7×7×64
   ├─ Conv 3×3, 128 filtres ─ BatchNorm ─ ReLU     →  7×7×128
   ├─ GlobalAveragePooling                          → 128
   ├─ Dropout 30 %                                  → 128
   └─ Dense 9, softmax                              → 9 probabilités (chiffres 1..9)
```

Rôle de chaque type de couche :

| Couche | Rôle |
|---|---|
| **Convolution 3×3** | Fait glisser de petits filtres 3×3 sur l'image. Chaque filtre apprend à repérer un motif : trait vertical, courbe, angle… Les couches suivantes combinent ces motifs en formes plus complexes (boucle du 6, barre du 7). |
| **BatchNormalization** | Recentre et remet à l'échelle les sorties de la convolution. L'entraînement devient plus stable et plus rapide. |
| **ReLU** | Non-linéarité `max(0, x)` : sans elle, empiler des couches reviendrait à une seule transformation linéaire. |
| **MaxPooling 2×2** | Garde le maximum de chaque carré 2×2 : divise la taille par 2 et rend le réseau tolérant aux petits décalages. |
| **GlobalAveragePooling** | Moyenne chacune des 128 cartes 7×7 en un seul nombre. On obtient un résumé de 128 valeurs de « quels motifs sont présents ». |
| **Dropout 30 %** | À l'entraînement uniquement, éteint au hasard 30 % des 128 valeurs. Le réseau ne peut plus dépendre d'une seule caractéristique, ce qui limite le sur-apprentissage. |
| **Dense + softmax** | Combine les 128 valeurs en 9 scores, transformés en probabilités qui somment à 1. |

Les convolutions n'ont pas de biais (`use_bias=False`) : la BatchNormalization
qui suit en a déjà un (β), il serait redondant.

**Champ récepteur.** Un neurone de la dernière convolution « voit » un carré de
**18×18 pixels** de l'image d'entrée (28×28). Il perçoit donc une grande partie
du chiffre, assez pour reconnaître sa forme globale.

## 3. Paramètres

Un paramètre est un nombre appris pendant l'entraînement (poids d'un filtre, biais…).

| Couche | Calcul | Entraînables | Non entraînables |
|---|---|---:|---:|
| Conv 1 | 3 × 3 × 1 × 32 | 288 | 0 |
| BatchNorm 1 | γ, β (entraînés) ; moyenne, variance glissantes | 64 | 64 |
| Conv 2 | 3 × 3 × 32 × 64 | 18 432 | 0 |
| BatchNorm 2 | 2 × 64 ; 2 × 64 | 128 | 128 |
| Conv 3 | 3 × 3 × 64 × 128 | 73 728 | 0 |
| BatchNorm 3 | 2 × 128 ; 2 × 128 | 256 | 256 |
| Dense | 128 × 9 poids + 9 biais | 1 161 | 0 |
| **Total** | | **94 057** | **448** |

**94 505 paramètres au total**, dont 94 057 appris par descente de gradient.
Les 448 non entraînables sont les moyennes et variances glissantes de la
BatchNormalization, mises à jour pendant l'entraînement mais pas par le gradient.

À titre de comparaison, c'est un **très petit** réseau : environ 270 fois moins
qu'un ResNet-50 (25,6 M). 78 % des paramètres sont dans la 3ᵉ convolution.

**Taille.** En float32, les poids pèsent 378 Ko. Le fichier `.keras` fait 1,2 Mo,
car il contient aussi l'état de l'optimiseur Adam (deux moyennes glissantes par
paramètre) et la description du modèle.

## 4. Nombre de neurones

Dans un CNN, un « neurone » correspond à une valeur de sortie d'une couche.

| Couche | Forme de sortie | Neurones |
|---|---|---:|
| Conv 1 | 28 × 28 × 32 | 25 088 |
| Conv 2 | 14 × 14 × 64 | 12 544 |
| Conv 3 | 7 × 7 × 128 | 6 272 |
| GlobalAveragePooling | 128 | 128 |
| Dense (sortie) | 9 | 9 |
| **Total (couches calculantes)** | | **44 041** |

Il y a bien plus de neurones que de paramètres par couche convolutive : les mêmes
poids d'un filtre sont réutilisés à chaque position de l'image (partage des poids).
C'est ce qui rend les CNN si économes.

**Coût de calcul.** Environ **7,45 millions de multiplications-additions** par
chiffre : 0,23 M pour la conv 1, 3,61 M pour la conv 2, 3,61 M pour la conv 3,
et 1 152 pour la couche dense. C'est négligeable pour un processeur moderne.

## 5. Données d'entraînement

| | |
|---|---|
| Dossier | `data/assets/` |
| Format | images JPEG 28×28 de chiffres **imprimés**, noir sur blanc, nommées `<chiffre>_<id>.jpg` |
| Nombre | 4 157 images (534 « 1 », 513 « 2 », 472 « 3 », 446 « 4 », 440 « 5 », 414 « 6 », 492 « 7 », 427 « 8 », 419 « 9 ») |
| Classes | 9 (chiffres 1 à 9). La classe 0 (case vide) a été retirée : les cases vides sont détectées avant le réseau (contraste de la case, voir `cells.is_blank_cell`). |
| Nettoyage | images floues ou sans contraste écartées dans `data/assets_trash/` (`scripts/auto_filter.py`) |
| Séparation | stratifiée : **3 533 images d'entraînement (85 %)**, **624 de validation (15 %)**, graine 42 |

Les classes sont équilibrées (de 414 à 534 images), il n'a donc pas été
nécessaire de pondérer les classes.

## 6. Entraînement (`scripts/train_cnn.py`)

| Réglage | Valeur | Pourquoi |
|---|---|---|
| Prétraitement | redimensionnement 28×28, pixels ÷ 255, inversion (chiffre blanc sur fond noir) | même convention que MNIST |
| Augmentation | rotation ±18°, translation ±5 %, zoom ±10 %, contraste ±20 % | simule des photos légèrement de travers ou mal éclairées |
| Fonction de perte | entropie croisée catégorielle | standard pour la classification |
| Optimiseur | Adam, taux d'apprentissage 10⁻³ | |
| Lot (batch) | 256 images, soit 14 pas par époque | |
| Époques | 20 au maximum | |
| ReduceLROnPlateau | divise le taux par 2 après 3 époques sans progrès (minimum 10⁻⁵) | affine en fin d'entraînement |
| EarlyStopping | arrête après 6 époques sans progrès de la perte de validation, **restaure les meilleurs poids** | évite le sur-apprentissage |
| Momentum BatchNorm | **0,9** au lieu de 0,99 par défaut | voir ci-dessous |

**Le piège du momentum BatchNorm.** Avec seulement 14 pas par époque, les
statistiques glissantes de la BatchNormalization n'avaient pas le temps de converger
avec le momentum par défaut (0,99). Le réseau apprenait bien (94 % sur
l'entraînement) mais s'effondrait en utilisation réelle : il répondait « 9 » à
presque tout (10 % de précision en validation, le hasard). Un momentum de 0,9
règle le problème.

L'entraînement prend environ 30 secondes sur un processeur.

## 7. Métriques

Mesurées sur les **624 images de validation**, jamais vues pendant l'apprentissage
des poids.

### Précision globale

| Métrique | Valeur |
|---|---|
| **Précision (accuracy)** | **94,7 %** (591 / 624) |
| Précision moyenne par classe (macro) | 95,2 % |
| Rappel moyen par classe (macro) | 94,6 % |
| F1 moyen (macro) | 94,6 % |

### Par chiffre

- La **précision** d'un chiffre est la part de bonnes réponses parmi les fois où le
  réseau a répondu ce chiffre.
- Le **rappel** d'un chiffre est la part de ce chiffre qui a bien été reconnue.

| Chiffre | Précision | Rappel | F1 | Images |
|---|---:|---:|---:|---:|
| 1 | 95,1 % | 97,5 % | 96,3 % | 80 |
| 2 | 96,2 % | 97,4 % | 96,8 % | 77 |
| 3 | 100 % | 85,9 % | 92,4 % | 71 |
| 4 | 83,8 % | 100 % | 91,2 % | 67 |
| 5 | 100 % | 90,9 % | 95,2 % | 66 |
| 6 | 86,1 % | 100 % | 92,5 % | 62 |
| 7 | 100 % | 98,6 % | 99,3 % | 74 |
| 8 | 95,2 % | 93,8 % | 94,5 % | 64 |
| 9 | 100 % | 87,3 % | 93,2 % | 63 |

### Matrice de confusion

Lignes = vrai chiffre, colonnes = chiffre prédit. La diagonale correspond aux bonnes réponses.

```
         1    2    3    4    5    6    7    8    9
  1     78    .    .    2    .    .    .    .    .
  2      .   75    .    2    .    .    .    .    .
  3      4    1   61    3    .    .    .    2    .
  4      .    .    .   67    .    .    .    .    .
  5      .    1    .    .   60    5    .    .    .
  6      .    .    .    .    .   62    .    .    .
  7      .    1    .    .    .    .   73    .    .
  8      .    .    .    1    .    3    .   60    .
  9      .    .    .    5    .    2    .    1   55
```

Ce qu'elle montre :
- **Le 4 et le 6 « attirent » les erreurs.** Le réseau répond 4 à tort 13 fois (des 1, 2, 3, 8 et 9)
  et 6 à tort 10 fois (des 5, 8 et 9). D'où leur rappel parfait mais leur précision plus faible.
- **Les confusions visuellement logiques** : 5 → 6 (5 fois), 9 → 4 (5 fois), 3 → 1 (4 fois).
- Le 7 est quasi parfait (1 seule erreur).

### Confiance et seuil `conf_min`

Le réseau donne une probabilité à sa réponse. Dans le pipeline, une réponse de
confiance inférieure à **0,6** (`predict.conf_min` dans `configs/default.yaml`) est
ignorée : la case est laissée vide plutôt que remplie avec un chiffre douteux.

| | |
|---|---|
| Confiance médiane | 0,85 |
| Confiance moyenne, bonnes réponses | 0,79 |
| Confiance moyenne, erreurs | 0,46 |
| Images sous le seuil 0,6 | **20,5 %** |
| **Précision des réponses acceptées (≥ 0,6)** | **98,8 %** |

Le seuil fait bien son travail : les erreurs ont une confiance nettement plus basse,
et en ne gardant que les réponses sûres, la précision passe de 94,7 % à 98,8 %.
En contrepartie, 1 image sur 5 est rejetée. Le solveur compense souvent, puisqu'une
case vide de plus reste en général résoluble. Dans l'interface, la case manquante
peut aussi être corrigée à la main.

### Sur de vraies grilles

Sur les 6 grilles d'exemple (`data/samples/`, 3 en mode clair et 3 en mode sombre),
**les 206 chiffres sont lus sans aucune erreur** (tests `tests/test_end_to_end.py`).
Les grilles imprimées sont plus nettes que les images d'entraînement, ce qui explique
un meilleur résultat qu'en validation.

## 8. Du chiffre de la case à l'entrée du réseau

Avant le réseau, chaque case passe par ces étapes (`cells.extract_digit`) :

1. **Case vide ?** Si le centre de la case est presque uniforme (écart < 40 niveaux de gris),
   la case est vide et le réseau n'est pas appelé.
2. **Binarisation** adaptative, puis recherche des taches d'encre (composantes connexes).
3. **Filtrage** : on rejette les taches trop fines, trop petites, décentrées ou qui
   traversent la case, qui sont en général des morceaux de lignes de la grille.
4. **Recadrage** de la tache retenue sur un carré, puis redimensionnement en 28×28.
5. **Normalisation** (`ocr/cnn.py`) : chiffre ramené à 20 px dans un carré de 28, blanc
   sur noir, pixels divisés par 255. Ces conventions sont lues dans `models/sudoku_cnn.meta.json` :
   ```json
   {"classes": [1,2,3,4,5,6,7,8,9], "input_range": "unit", "polarity": "white_on_black"}
   ```
   `classes` fait correspondre chaque sortie du réseau à un chiffre. Sans ce fichier,
   la sortie n° 0 serait prise pour un « 0 » (case vide) au lieu d'un « 1 ».

En cas de doute, la case est relue avec trois binarisations différentes et les réponses
sont départagées par un vote (`pipeline._ocr_cell_multi`).

## 9. Limites connues

- **Le réseau manque d'assurance.** Une confiance médiane de 0,85 est basse pour un
  problème aussi simple, et 20 % des chiffres tombent sous le seuil. Un entraînement
  plus long ou avec moins de dropout améliorerait sans doute ce point.
- **La validation sert aussi à arrêter l'entraînement.** EarlyStopping et la sauvegarde
  du meilleur modèle regardent la perte de validation, donc les 94,7 % sont légèrement
  optimistes. Un jeu de test séparé, jamais utilisé à l'entraînement, donnerait une mesure
  plus honnête.
- **Écart entre entraînement et utilisation.** À l'entraînement, les images sont en
  niveaux de gris lissés. En utilisation, le chiffre est binarisé (noir ou blanc) puis
  redimensionné. Les deux se ressemblent mais ne sont pas identiques.
- **Chiffres imprimés uniquement.** Aucune écriture manuscrite dans les données : une
  grille remplie à la main serait mal lue.
- **Grilles 9×9 uniquement.** Les mini-sudokus 6×6 (LinkedIn…) ne sont pas pris en charge.
- **Lenteur évitable.** Le pipeline interroge le réseau case par case : environ 28 ms par
  appel, donc jusqu'à 2 à 3 s pour une grille pleine. Un seul appel pour toutes les cases
  prendrait environ 50 ms au total.

## 10. Reproduire ces mesures

```bash
make install                   # environnement avec les extras train + dev
make train                     # réentraîne models/sudoku_cnn.keras (+ .meta.json)
python scripts/debug_val.py    # précision, confiance, rapport par chiffre et matrice de confusion
make test                      # dont la lecture exacte des 6 grilles d'exemple
```

Architecture et nombre de paramètres :

```python
from tensorflow import keras
keras.models.load_model("models/sudoku_cnn.keras", compile=False).summary()
```

Réentraîner produit un modèle légèrement différent : initialisation et augmentation
sont aléatoires, donc les métriques varient d'environ ±1 à 2 points d'un entraînement
à l'autre.
