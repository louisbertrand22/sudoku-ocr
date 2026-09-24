"""Chargement de la configuration YAML (configs/default.yaml) avec valeurs par défaut."""
from __future__ import annotations
import copy
import os
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "configs/default.yaml"

# Référence des clés acceptées : toute clé absente d'ici est refusée (faute de frappe)
DEFAULTS: dict[str, dict[str, Any]] = {
    "ocr": {
        "backend": "cnn",                          # "cnn" | "tesseract"
        "cnn_weights": "models/mnist_cnn.keras",
    },
    "detect": {
        "warp_size": 450,
    },
    "predict": {
        "conf_min": 0.6,
    },
    "overlay": {
        "color": [120, 120, 120],                  # BGR des chiffres trouvés
        "given_color": [120, 120, 120],            # BGR des chiffres de départ
        "scale": 1.2,
        "thickness": 2,
        "show_mode": "all",                        # "all" | "new" | "givens_only"
    },
}

_CHOICES = {
    ("ocr", "backend"): {"cnn", "tesseract"},
    ("overlay", "show_mode"): {"all", "new", "givens_only"},
}


def _merge(cfg: dict, override: dict, origin: str) -> None:
    for section, values in override.items():
        if section not in DEFAULTS:
            raise ValueError(f"{origin}: section inconnue '{section}' "
                             f"(attendu : {', '.join(DEFAULTS)})")
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"{origin}: la section '{section}' doit être un dictionnaire")
        for key, value in values.items():
            if key not in DEFAULTS[section]:
                raise ValueError(f"{origin}: clé inconnue '{section}.{key}' "
                                 f"(attendu : {', '.join(DEFAULTS[section])})")
            cfg[section][key] = value


def _validate(cfg: dict) -> None:
    for (section, key), allowed in _CHOICES.items():
        if cfg[section][key] not in allowed:
            raise ValueError(f"{section}.{key} = {cfg[section][key]!r} invalide "
                             f"(attendu : {', '.join(sorted(allowed))})")
    for key in ("color", "given_color"):
        col = cfg["overlay"][key]
        if not (isinstance(col, (list, tuple)) and len(col) == 3):
            raise ValueError(f"overlay.{key} doit être une liste [B, G, R]")


def load_config(paths: str | list[str] | None = None, overrides: dict | None = None) -> dict:
    """Valeurs par défaut <- fichiers YAML dans l'ordre donné <- overrides (ex. options CLI).

    Chaque fichier ne remplace que les clés qu'il contient.
    """
    if isinstance(paths, str):
        paths = [paths]
    cfg = copy.deepcopy(DEFAULTS)
    for path in paths or []:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config introuvable: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: la config doit être un dictionnaire YAML")
        _merge(cfg, data, path)
    if overrides:
        _merge(cfg, overrides, "options")
    _validate(cfg)
    return cfg
