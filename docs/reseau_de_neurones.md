# Le réseau de neurones de sudoku-ocr

> Toutes les valeurs de ce document ont été mesurées sur le modèle versionné
> `models/sudoku_cnn.keras` (entraîné par `make train`, recette E3 de la section 7)
> et sur un jeu de test de 624 images jamais vues à l'entraînement. Les commandes
> pour les reproduire sont en fin de document.

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

Séparation stratifiée en trois jeux, chacun gardant la proportion de chaque chiffre :

| Jeu | Images | Sert à |
|---|---:|---|
| Entraînement | 3 003 | apprendre les poids |
| Arrêt anticipé | 530 | décider quand arrêter l'entraînement |
| **Test** | **624** | **mesurer, uniquement** : jamais utilisé pendant l'entraînement (graine 42) |

Les classes sont équilibrées (de 414 à 534 images), il n'a donc pas été
nécessaire de pondérer les classes.

## 6. Entraînement (`scripts/train_cnn.py`)

### Idée principale : apprendre sur ce que le réseau verra vraiment

Dans l'app, un chiffre n'arrive jamais « brut » au réseau : il est d'abord extrait
de sa case, binarisé, recadré puis réduit à 20 px dans un carré de 28 (section 9).
L'entraînement reproduit exactement ce chemin :

1. chaque image 28×28 est agrandie en case de 56×56, la taille d'une vraie case ;
2. **5 versions abîmées** sont créées par image : rotation ±8°, zoom ±15 %, décalage
   ±3 px, trait plus épais ou plus fin, flou, contraste réduit ou fond grisé, bruit,
   compression JPEG ;
3. l'original et ses 5 variantes passent par **le prétraitement de l'app**
   (`extract_digit` puis `to_28x28_white_on_black`).

On obtient **17 797 images d'entraînement**. Une poignée est rejetée par l'extraction,
comme elle le serait dans l'app.

### Réglages

| Réglage | Valeur | Pourquoi |
|---|---|---|
| Fonction de perte | entropie croisée catégorielle | standard pour la classification |
| Optimiseur | Adam, taux d'apprentissage 10⁻³ | |
| Lot (batch) | 64 images, soit environ 280 pas par époque | |
| Époques | 60 au maximum (25 en pratique) | |
| ReduceLROnPlateau | divise le taux par 2 après 5 époques sans progrès (minimum 10⁻⁵) | affine en fin d'entraînement |
| EarlyStopping | arrête après 10 époques sans progrès sur le jeu d'arrêt anticipé, **restaure les meilleurs poids** | évite le sur-apprentissage |
| Momentum BatchNorm | **0,9** au lieu de 0,99 par défaut | voir ci-dessous |
| Graine | 1 | entraînement reproductible |

**Le piège du momentum BatchNorm.** Avec peu de pas par époque, les statistiques
glissantes de la BatchNormalization n'avaient pas le temps de converger avec le
momentum par défaut (0,99). Le réseau apprenait bien (94 % sur l'entraînement) mais
s'effondrait en utilisation réelle : il répondait « 9 » à presque tout (10 % de
précision en validation, le hasard). Un momentum de 0,9 règle le problème.

L'entraînement prend environ **2 minutes** sur un processeur (`make train`).

## 7. Expériences : comment la recette a été choisie

`scripts/ocr_experiments.py` entraîne plusieurs recettes sur les mêmes données, avec
2 graines chacune, et les mesure sur le même jeu de test **avec le prétraitement de
l'app**. Une image que l'extraction rejette compte comme une erreur, et 9 des 624
images sont dans ce cas : **la précision maximale atteignable est donc 98,6 %**.

La mesure la plus importante est le nombre de **chiffres faux acceptés** : une
réponse fausse mais assez sûre d'elle (confiance ≥ 0,6) pour être inscrite dans la
grille. Un seul suffit à rendre la grille insoluble ou la solution fausse. Une
réponse rejetée laisse seulement une case vide, que le solveur comble souvent.

| Recette | Précision (test app) | Couverture | Chiffres faux acceptés |
|---|---:|---:|---:|
| E0 : ancien modèle (niveaux de gris, lots de 256, 20 époques) | 96,3 % | 91,3 % | 4 |
| E1 : ancienne recette, réentraînée | 95,7 à 95,8 % | 94 à 95 % | 6 à 8 |
| E2 : ancienne recette, plus longue (lots de 64, 60 époques) | 95,5 à 96,5 % | 96 à 98 % | 7 à 11 |
| **E3 : prétraitement de l'app + images abîmées (retenue)** | **97,8 à 98,1 %** | **98 %** | **1 à 3** |
| E4 : E3 + 5 400 chiffres synthétiques (22 polices système) | 97,8 à 98,1 % | 98 % | 3 à 4 |

*Couverture : part des chiffres lus avec une confiance ≥ 0,6 (les autres laissent la case vide).*

Enseignements :
- **Aligner l'entraînement sur l'app est le vrai gain.** E3 frôle le plafond de 98,6 %
  avec les deux graines.
- **Entraîner plus longtemps est un piège.** E2 atteint 99,5 % sur le test en niveaux
  de gris, mais inscrit *plus* de chiffres faux dans les conditions réelles.
- **Les chiffres synthétiques n'apportent rien de mesurable ici** et rendent
  l'entraînement non reproductible : le résultat dépend des polices installées sur
  la machine. E3 est donc préféré à E4.
- Sur les 206 cases des grilles d'exemple et sur 10 chiffres d'une capture LinkedIn
  (police jamais vue, mode sombre), **tous les modèles font un sans-faute**. Ces tests
  sont trop faciles pour départager les recettes.
- **Nuance :** le modèle livré vient de la graine 1, qui est la meilleure des deux
  graines de E3 sur ce test. Le chiffre de 98,1 % est donc légèrement flatteur. La
  graine 0 donne 97,8 % et 3 chiffres faux acceptés.

## 8. Métriques du modèle livré

Mesurées sur les **624 images de test**, avec le prétraitement de l'app.

### Précision globale

| Métrique | Ancien modèle | **Modèle livré** |
|---|---:|---:|
| Précision (rejets de l'extraction comptés comme erreurs) | 96,3 % | **98,1 %** |
| Précision sur les 615 images extraites | 97,7 % | **99,5 %** |
| Confiance médiane | 0,94 | **1,00** |
| Couverture (confiance ≥ 0,6) | 91,3 % | **97,9 %** |
| Précision des réponses acceptées | 99,3 % | **99,8 %** |
| Chiffres faux acceptés | 4 | **1** |

### Par chiffre (sur les 615 images extraites)

- La **précision** d'un chiffre est la part de bonnes réponses parmi les fois où le
  réseau a répondu ce chiffre.
- Le **rappel** d'un chiffre est la part de ce chiffre qui a bien été reconnue.

| Chiffre | Précision | Rappel | F1 | Images |
|---|---:|---:|---:|---:|
| 1 | 97,5 % | 100 % | 98,8 % | 79 |
| 2 | 100 % | 98,7 % | 99,3 % | 75 |
| 3 | 100 % | 98,6 % | 99,3 % | 71 |
| 4 | 100 % | 100 % | 100 % | 63 |
| 5 | 100 % | 100 % | 100 % | 65 |
| 6 | 98,4 % | 100 % | 99,2 % | 61 |
| 7 | 100 % | 98,6 % | 99,3 % | 74 |
| 8 | 100 % | 100 % | 100 % | 64 |
| 9 | 100 % | 100 % | 100 % | 63 |

### Matrice de confusion

Lignes = vrai chiffre, colonnes = chiffre prédit. La diagonale correspond aux bonnes réponses.

```
         1    2    3    4    5    6    7    8    9
  1     79    .    .    .    .    .    .    .    .
  2      1   74    .    .    .    .    .    .    .
  3      1    .   70    .    .    .    .    .    .
  4      .    .    .   63    .    .    .    .    .
  5      .    .    .    .   65    .    .    .    .
  6      .    .    .    .    .   61    .    .    .
  7      .    .    .    .    .    1   73    .    .
  8      .    .    .    .    .    .    .   64    .
  9      .    .    .    .    .    .    .    .   63
```

Il ne reste que **3 erreurs sur 615** : un 2 et un 3 lus « 1 », et un 7 lu « 6 ».
Les confusions typiques de l'ancien modèle (réponses « 4 » et « 6 » données à tort,
5 → 6, 9 → 4, mesurées sur le test en niveaux de gris) ont disparu.

### Confiance et seuil `conf_min`

Le réseau donne une probabilité à sa réponse. Dans le pipeline, une réponse de
confiance inférieure à **0,6** (`predict.conf_min` dans `configs/default.yaml`) est
ignorée : la case est laissée vide plutôt que remplie avec un chiffre douteux.

L'ancien modèle manquait d'assurance (confiance médiane 0,94 dans les conditions de
l'app, et même 0,85 en niveaux de gris bruts). Il rejetait ainsi près d'un chiffre sur
dix. Le modèle livré est sûr de lui (médiane 1,00) : **97,9 %** des chiffres sont
acceptés, et une seule réponse fausse passe le seuil.

### Sur de vraies grilles

Sur les 6 grilles d'exemple (`data/samples/`, 3 en mode clair et 3 en mode sombre),
**les 206 chiffres sont lus sans aucune erreur** (tests `tests/test_end_to_end.py`).

## 9. Du chiffre de la case à l'entrée du réseau

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

L'entraînement suit exactement ces mêmes étapes (section 6).

En cas de doute, la case est relue avec trois binarisations différentes et les réponses
sont départagées par un vote (`pipeline._ocr_cell_multi`).

## 10. Limites connues

- **1,4 % des chiffres sont perdus avant le réseau.** 9 images de test sur 624 sont
  rejetées par l'extraction (tache jugée trop fine ou décentrée). Elles sont désormais
  la première source d'erreur, devant le réseau lui-même.
- **Chiffres imprimés uniquement.** Aucune écriture manuscrite dans les données : une
  grille remplie à la main serait mal lue.
- **Grilles 9×9 uniquement.** Les mini-sudokus 6×6 (LinkedIn…) ne sont pas pris en charge,
  même si leurs chiffres sont bien lus (section 7).
- **Tests sur de vraies photos encore rares.** Les mesures reposent sur un seul jeu de
  données imprimé et 6 grilles d'exemple. Des photos prises en biais ou mal éclairées
  restent à mesurer.
- **Lenteur évitable.** Le pipeline interroge le réseau case par case : environ 28 ms par
  appel, donc jusqu'à 2 à 3 s pour une grille pleine. Un seul appel pour toutes les cases
  prendrait environ 50 ms au total.

## 11. Reproduire ces mesures

```bash
make install                        # environnement avec les extras train + dev
make train                          # réentraîne models/sudoku_cnn.keras (+ .meta.json), ~2 min
python scripts/debug_val.py         # métriques, rapport par chiffre et matrice de confusion (test)
python scripts/dump_preprocessed.py # exemples d'entraînement tels que le réseau les voit
python scripts/ocr_experiments.py   # compare les recettes E0 à E4 (2 graines, ~15 min)
make test                           # dont la lecture exacte des 6 grilles d'exemple
```

Architecture et nombre de paramètres :

```python
from tensorflow import keras
keras.models.load_model("models/sudoku_cnn.keras", compile=False).summary()
```

`make train` utilise la graine 1 et redonne exactement le modèle livré sur la même
machine. Avec une autre graine, les métriques varient d'environ ±0,3 point.
