import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from tensorflow import keras
from tensorflow.keras import layers

from sudoku_ocr.ocr.cnn import CNNOCR, save_meta


def _fixed_model(n_out: int, winner: int, rescaling: bool, confidence: float = 10.0):
    """Modèle dont la sortie ne dépend pas de l'entrée : l'index `winner` gagne toujours."""
    inp = layers.Input((28, 28, 1))
    x = layers.Rescaling(1.0 / 255.0)(inp) if rescaling else inp
    x = layers.Flatten()(x)
    bias = np.zeros(n_out, dtype=np.float32)
    bias[winner] = confidence
    out = layers.Dense(n_out, activation="softmax", kernel_initializer="zeros",
                       bias_initializer=keras.initializers.Constant(bias))(x)
    return keras.Model(inp, out)


def _save(model, tmp_path, name="model.keras") -> str:
    path = str(tmp_path / name)
    model.save(path)
    return path


def _digit_img():
    img = np.zeros((28, 28), dtype=np.uint8)
    img[6:22, 12:16] = 255  # un "1" blanc sur fond noir
    return img


def test_sudoku_model_without_meta_maps_first_output_to_digit_1(tmp_path):
    # modèle issu de train_cnn.py sans classe 0 : 9 sorties, index 0 == chiffre 1
    path = _save(_fixed_model(9, winner=0, rescaling=False), tmp_path)
    ocr = CNNOCR(weights_path=path, train_if_missing=False)
    assert ocr.classes == list(range(1, 10))
    assert ocr.meta["input_range"] == "unit"
    assert ocr.predict_digit(_digit_img()) == 1


def test_sudoku_model_input_is_scaled_to_unit_range(tmp_path):
    path = _save(_fixed_model(9, winner=4, rescaling=False), tmp_path)
    ocr = CNNOCR(weights_path=path, train_if_missing=False)
    x = ocr._prepare(_digit_img())
    assert x.shape == (1, 28, 28, 1)
    assert 0.9 < x.max() <= 1.0
    assert ocr.predict_digit(_digit_img()) == 5


def test_mnist_model_without_meta_keeps_byte_range_and_blank_class(tmp_path):
    path = _save(_fixed_model(10, winner=0, rescaling=True), tmp_path)
    ocr = CNNOCR(weights_path=path, train_if_missing=False)
    assert ocr.meta["input_range"] == "byte"
    assert ocr._prepare(_digit_img()).max() == 255.0
    assert ocr.predict_digit(_digit_img()) == 0  # classe 0 == case vide


def test_meta_file_overrides_guess(tmp_path):
    path = _save(_fixed_model(9, winner=0, rescaling=False), tmp_path)
    save_meta(path, {"classes": list(range(9, 0, -1)), "input_range": "unit",
                     "polarity": "white_on_black"})
    ocr = CNNOCR(weights_path=path, train_if_missing=False)
    assert ocr.predict_digit(_digit_img()) == 9


def test_black_on_white_polarity_inverts_input(tmp_path):
    path = _save(_fixed_model(9, winner=0, rescaling=False), tmp_path)
    save_meta(path, {"classes": list(range(1, 10)), "input_range": "unit",
                     "polarity": "black_on_white"})
    ocr = CNNOCR(weights_path=path, train_if_missing=False)
    x = ocr._prepare(_digit_img())
    assert x[0, 0, 0, 0] == 1.0  # fond noir devenu blanc


def test_low_confidence_returns_blank(tmp_path):
    path = _save(_fixed_model(9, winner=3, rescaling=False, confidence=0.1), tmp_path)
    ocr = CNNOCR(weights_path=path, train_if_missing=False, conf_min=0.6)
    assert ocr.predict_digit(_digit_img()) == 0


def test_meta_class_count_mismatch_raises(tmp_path):
    path = _save(_fixed_model(9, winner=0, rescaling=False), tmp_path)
    save_meta(path, {"classes": list(range(10)), "input_range": "unit"})
    with pytest.raises(ValueError):
        CNNOCR(weights_path=path, train_if_missing=False)


def test_missing_model_without_training_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CNNOCR(weights_path=str(tmp_path / "absent.keras"), train_if_missing=False)
