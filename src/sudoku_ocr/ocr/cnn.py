# CNN OCR implementation
from __future__ import annotations
import json
import os
import numpy as np

# TensorFlow (optionnel) avec garde robuste
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:  # pragma: no cover
    TF_AVAILABLE = False
    keras = None   # type: ignore
    layers = None  # type: ignore

from .base import OCRBase, to_28x28_white_on_black, postprocess_digit


# ------------------------- métadonnées du modèle ------------------------- #
#
# Chaque modèle est accompagné d'un fichier "<modele>.meta.json" qui décrit
# le contrat d'entrée/sortie utilisé à l'entraînement :
#   - classes      : chiffre associé à chaque sortie du softmax (index -> chiffre)
#   - input_range  : "byte" (pixels 0..255) ou "unit" (pixels 0..1)
#   - polarity     : "white_on_black" ou "black_on_white"

MNIST_META = {"classes": list(range(10)), "input_range": "byte", "polarity": "white_on_black"}


def meta_path_for(model_path: str) -> str:
    return os.path.splitext(model_path)[0] + ".meta.json"


def save_meta(model_path: str, meta: dict) -> str:
    path = meta_path_for(model_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def load_meta(model_path: str) -> dict | None:
    path = meta_path_for(model_path)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _guess_meta(model) -> dict:
    """Déduit les métadonnées d'un modèle sans fichier .meta.json (modèles antérieurs).

    - 10 sorties -> chiffres 0..9 ; 9 sorties -> chiffres 1..9 (classe 0 retirée)
    - couche Rescaling présente -> entrée 0..255, sinon entrée 0..1
    """
    n_out = int(model.output_shape[-1])
    if n_out == 10:
        classes = list(range(10))
    elif n_out == 9:
        classes = list(range(1, 10))
    else:
        raise ValueError(
            f"Impossible de deviner les classes d'un modèle à {n_out} sorties : "
            f"fournissez un fichier .meta.json."
        )
    has_rescaling = any(isinstance(l, layers.Rescaling) for l in model.layers)
    return {
        "classes": classes,
        "input_range": "byte" if has_rescaling else "unit",
        "polarity": "white_on_black",
    }


def _build_cnn():
    """Construit un petit CNN pour MNIST (28x28x1)."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow n'est pas installé.")
    model = keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Rescaling(1.0 / 255.0),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax'),  # classes 0..9
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


class CNNOCR(OCRBase):
    """
    Backend OCR basé sur un CNN Keras.

    Charge n'importe quel modèle complet (.keras) : le CNN MNIST par défaut ou
    un modèle produit par scripts/train_cnn.py. Le prétraitement et le mapping
    des sorties sont pilotés par le fichier .meta.json associé.

    Args:
        weights_path: chemin du modèle Keras à charger/sauver
        train_if_missing: entraîne sur MNIST si le modèle n'existe pas
        epochs: nb d'époques pour l'entraînement initial
        batch_size: batch size d'entraînement
        conf_min: seuil de confiance pour accepter la prédiction
    """

    def __init__(self,
                 weights_path: str = "models/mnist_cnn.keras",
                 train_if_missing: bool = True,
                 epochs: int = 3,
                 batch_size: int = 128,
                 conf_min: float = 0.6):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow n'est pas installé. Installez-le ou utilisez le backend Tesseract.")
        self.weights_path = weights_path
        self.train_if_missing = train_if_missing
        self.epochs = epochs
        self.batch_size = batch_size
        self.conf_min = conf_min

        self.model, self.meta = self._load_or_train()
        self.classes = [int(c) for c in self.meta["classes"]]
        if len(self.classes) != int(self.model.output_shape[-1]):
            raise ValueError(
                f"{meta_path_for(weights_path)} déclare {len(self.classes)} classes "
                f"mais le modèle a {self.model.output_shape[-1]} sorties."
            )

    # -------------------------- lifecycle -------------------------- #
    def _load_or_train(self):
        if os.path.exists(self.weights_path):
            model = keras.models.load_model(self.weights_path, compile=False)
            meta = load_meta(self.weights_path) or _guess_meta(model)
            return model, meta
        if not self.train_if_missing:
            raise FileNotFoundError(
                f"Modèle introuvable: {self.weights_path} "
                f"(entraînez-le avec scripts/train_cnn.py)"
            )
        self.train_from_mnist(self.weights_path, self.epochs, self.batch_size)
        return keras.models.load_model(self.weights_path, compile=False), dict(MNIST_META)

    # --------------------------- inference -------------------------- #
    def _prepare(self, img28: np.ndarray) -> np.ndarray:
        """Applique le même prétraitement qu'à l'entraînement -> tenseur (1,28,28,1)."""
        x = to_28x28_white_on_black(img28).astype('float32')
        if self.meta.get("polarity", "white_on_black") == "black_on_white":
            x = 255.0 - x
        if self.meta.get("input_range", "byte") == "unit":
            x = x / 255.0
        return x[np.newaxis, ..., np.newaxis]

    def predict_digit(self, img28: np.ndarray) -> int:
        if img28 is None:
            return 0
        probs = self.model.predict(self._prepare(img28), verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        digit = self.classes[idx]
        if digit == 0 or conf < self.conf_min:
            return 0  # case vide ou prédiction incertaine
        return postprocess_digit(digit)

    # -------------------------- utilitaires ------------------------- #
    @staticmethod
    def train_from_mnist(weights_out: str = "models/mnist_cnn.keras",
                         epochs: int = 3,
                         batch_size: int = 128) -> str:
        """(Ré)entraîne rapidement sur MNIST et sauvegarde le modèle + métadonnées."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow n'est pas installé.")
        model = _build_cnn()
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  epochs=epochs, batch_size=batch_size, verbose=2)
        os.makedirs(os.path.dirname(weights_out) or ".", exist_ok=True)
        model.save(weights_out)
        save_meta(weights_out, MNIST_META)
        return weights_out
