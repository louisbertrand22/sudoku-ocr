# CNN OCR implementation
from __future__ import annotations
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
    Backend OCR basé sur un CNN entraîné sur MNIST.

    Args:
        weights_path: chemin des poids Keras à charger/sauver
        train_if_missing: entraîne sur MNIST si les poids n'existent pas
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

        self.model = _build_cnn()
        self._ensure_weights()

    # -------------------------- lifecycle -------------------------- #
    def _ensure_weights(self):
        # charge les poids s'ils existent, sinon entraîne vite sur MNIST
        if os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)
            return
        if not self.train_if_missing:
            raise FileNotFoundError(f"Poids introuvables: {self.weights_path}")
        self._train_and_save()

    def _train_and_save(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=2,
        )
        os.makedirs(os.path.dirname(self.weights_path) or ".", exist_ok=True)
        # on sauvegarde le modèle complet (format .keras)
        self.model.save(self.weights_path)

    # --------------------------- inference -------------------------- #
    def predict_digit(self, img28: np.ndarray) -> int:
        if img28 is None:
            return 0
        x = to_28x28_white_on_black(img28).astype('float32')[np.newaxis, ..., np.newaxis]
        probs = self.model.predict(x, verbose=0)[0]
        cls = int(np.argmax(probs))    # 0..9
        conf = float(np.max(probs))
        if cls == 0:
            return 0  # Sudoku n'utilise pas 0
        return postprocess_digit(cls if conf >= self.conf_min else 0)

    # -------------------------- utilitaires ------------------------- #
    @staticmethod
    def train_from_mnist(weights_out: str = "models/mnist_cnn.keras",
                         epochs: int = 3,
                         batch_size: int = 128) -> str:
        """(Ré)entraîne rapidement et sauvegarde des poids."""
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
        return weights_out