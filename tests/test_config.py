from pathlib import Path

import pytest

from sudoku_ocr import cli
from sudoku_ocr.config import DEFAULTS, load_config

ROOT = Path(__file__).resolve().parent.parent


def _write(tmp_path, text: str) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_defaults_without_file():
    assert load_config() == DEFAULTS
    assert load_config() is not DEFAULTS  # copie, pas l'original


def test_yaml_overrides_defaults_and_keeps_others(tmp_path):
    cfg = load_config(_write(tmp_path, "ocr:\n  cnn_weights: m.keras\npredict:\n  conf_min: 0.8\n"))
    assert cfg["ocr"]["cnn_weights"] == "m.keras"
    assert cfg["ocr"]["backend"] == "cnn"
    assert cfg["predict"]["conf_min"] == 0.8
    assert cfg["detect"]["warp_size"] == 450


def test_overrides_win_over_yaml(tmp_path):
    cfg = load_config(_write(tmp_path, "ocr:\n  backend: cnn\n"), {"ocr": {"backend": "tesseract"}})
    assert cfg["ocr"]["backend"] == "tesseract"


def test_empty_yaml_gives_defaults(tmp_path):
    assert load_config(_write(tmp_path, "")) == DEFAULTS


@pytest.mark.parametrize("text", [
    "ocr:\n  cnn_weigths: m.keras\n",      # faute de frappe dans une clé
    "detection:\n  warp_size: 300\n",       # section inconnue
    "ocr:\n  backend: easyocr\n",           # valeur hors choix
    "overlay:\n  show_mode: everything\n",
    "overlay:\n  color: [1, 2]\n",
    "- a\n- b\n",                           # pas un dictionnaire
])
def test_invalid_config_is_rejected(tmp_path, text):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, text))


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "absent.yaml"))


def test_repo_default_yaml_is_valid():
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    assert cfg["ocr"]["cnn_weights"] == "models/sudoku_cnn.keras"


def test_cli_reads_explicit_config_and_flags_override(tmp_path):
    path = _write(tmp_path, "ocr:\n  cnn_weights: from_yaml.keras\npredict:\n  conf_min: 0.9\n")
    args = cli.build_parser().parse_args(
        ["--image", "x.png", "--config", path, "--weights", "from_cli.keras"])
    cfg = cli.config_from_args(args)
    assert cfg["ocr"]["cnn_weights"] == "from_cli.keras"
    assert cfg["predict"]["conf_min"] == 0.9


def test_cli_uses_repo_default_config(monkeypatch):
    monkeypatch.chdir(ROOT)
    args = cli.build_parser().parse_args(["--image", "x.png"])
    assert cli.config_from_args(args)["ocr"]["cnn_weights"] == "models/sudoku_cnn.keras"


def test_cli_falls_back_to_defaults_outside_repo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = cli.build_parser().parse_args(["--image", "x.png", "--backend", "tesseract"])
    cfg = cli.config_from_args(args)
    assert cfg["ocr"]["backend"] == "tesseract"
    assert cfg["ocr"]["cnn_weights"] == DEFAULTS["ocr"]["cnn_weights"]


def test_cli_invalid_config_exits_with_usage_error(tmp_path, capsys):
    path = _write(tmp_path, "ocr:\n  cnn_weigths: m.keras\n")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--image", "x.png", "--config", path])
    assert exc.value.code == 2
    assert "cnn_weigths" in capsys.readouterr().err
