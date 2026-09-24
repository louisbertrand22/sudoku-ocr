"""Interface Streamlit : image -> grille lue -> correction manuelle -> solution.

Lancement : sudoku-ocr-ui   (ou : streamlit run src/sudoku_ocr/ui/app.py)
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import streamlit as st

from sudoku_ocr.config import DEFAULT_CONFIG_PATH, load_config
from sudoku_ocr.pipeline import _make_ocr_backend, read_grid, render_solution
from sudoku_ocr.solver import count_solutions, find_conflicts, solve
from sudoku_ocr.ui.helpers import (
    COLUMNS, bgr_to_hex, decode_image, draw_detection, draw_reading, encode_png,
    frame_to_grid, grid_to_frame, hex_to_bgr, to_rgb,
)

# chemins relatifs (configs/, models/, data/samples/) résolus depuis la racine du
# dépôt, quel que soit le dossier de lancement (Streamlit Cloud, autre dossier)
REPO_ROOT = Path(__file__).resolve().parents[3]
if (REPO_ROOT / "configs").is_dir():
    os.chdir(REPO_ROOT)

SAMPLES_DIR = Path("data/samples")
SHOW_MODES = {"all": "Tout", "new": "Chiffres trouvés", "givens_only": "Chiffres de départ"}


# ------------------------------------------------------------------ cache -- #

@st.cache_resource(show_spinner="Chargement du modèle…")
def _load_ocr(backend: str, weights: str, conf_min: float):
    cfg = {"ocr": {"backend": backend, "cnn_weights": weights}, "predict": {"conf_min": conf_min}}
    return _make_ocr_backend(backend, load_config(overrides=cfg))


@st.cache_data(show_spinner="Lecture de la grille…", max_entries=16)
def _read(image_bytes: bytes, backend: str, weights: str, conf_min: float, warp_size: int):
    img = decode_image(image_bytes)
    cfg = {"ocr": {"backend": backend, "cnn_weights": weights},
           "predict": {"conf_min": conf_min}, "detect": {"warp_size": warp_size}}
    return read_grid(img, cfg, _load_ocr(backend, weights, conf_min))


# ---------------------------------------------------------------- sidebar -- #

def _sidebar(base: dict) -> dict:
    sb = st.sidebar
    sb.header("Réglages")

    sb.subheader("Lecture")
    backend = sb.radio("Moteur OCR", ["cnn", "tesseract"],
                       index=["cnn", "tesseract"].index(base["ocr"]["backend"]),
                       format_func={"cnn": "CNN", "tesseract": "Tesseract"}.get, horizontal=True)
    weights = sb.text_input("Modèle CNN", base["ocr"]["cnn_weights"], disabled=backend != "cnn")
    conf_min = sb.slider("Confiance minimale", 0.0, 1.0, float(base["predict"]["conf_min"]), 0.05,
                         help="En dessous, un chiffre lu est ignoré (case laissée vide).")

    sb.subheader("Affichage")
    show_mode = sb.segmented_control("Chiffres réincrustés", list(SHOW_MODES),
                                     default=base["overlay"]["show_mode"],
                                     format_func=SHOW_MODES.get) or base["overlay"]["show_mode"]
    c1, c2 = sb.columns(2)
    color = c1.color_picker("Trouvés", bgr_to_hex(base["overlay"]["color"]))
    given_color = c2.color_picker("Départ", bgr_to_hex(base["overlay"]["given_color"]))
    scale = sb.slider("Taille des chiffres", 0.5, 2.5, float(base["overlay"]["scale"]), 0.1)

    return load_config(overrides={
        "ocr": {"backend": backend, "cnn_weights": weights},
        "predict": {"conf_min": conf_min},
        "detect": {"warp_size": base["detect"]["warp_size"]},
        "overlay": {**base["overlay"], "show_mode": show_mode, "scale": scale,
                    "color": hex_to_bgr(color), "given_color": hex_to_bgr(given_color)},
    })


# ----------------------------------------------------------------- source -- #

def _pick_image() -> tuple[str, bytes] | None:
    samples = sorted(SAMPLES_DIR.glob("*.png")) + sorted(SAMPLES_DIR.glob("*.jpg"))
    sources = (["Exemple"] if samples else []) + ["Importer une image"]
    source = st.segmented_control("Source", sources, default=sources[0],
                                  label_visibility="collapsed") or sources[0]
    if source == "Exemple":
        path = st.selectbox("Image d'exemple", samples, format_func=lambda p: p.name)
        return path.name, path.read_bytes()
    up = st.file_uploader("Photo ou capture d'une grille", type=["png", "jpg", "jpeg", "webp", "bmp"])
    if up is None:
        return None
    return up.name, up.getvalue()


# ------------------------------------------------------------------ status -- #

def _cells(cells: list[tuple[int, int]]) -> str:
    return ", ".join(f"L{r + 1}C{c + 1}" for r, c in cells)


def _status(grid: np.ndarray, conflicts: list[tuple[int, int]], n_solutions: int) -> None:
    n = int(np.count_nonzero(grid))
    st.metric("Chiffres dans la grille", n)
    if conflicts:
        st.error(f"Chiffres en conflit : {_cells(conflicts)}. Corrigez-les dans la grille.")
    elif n_solutions == 0:
        st.error("Aucune solution : un chiffre est probablement mal lu.")
    elif n_solutions > 1:
        st.warning("Plusieurs solutions possibles : un chiffre a probablement été manqué "
                   "(une grille valide en a au moins 17).")
    else:
        st.success("Grille valide, solution unique.")


# ------------------------------------------------------------------- page -- #

def main() -> None:
    st.set_page_config(page_title="Sudoku OCR", page_icon="🔢", layout="wide")
    st.title("Sudoku OCR")
    st.caption("Détecte la grille, lit les chiffres, vous laisse corriger, puis résout.")

    base = load_config(DEFAULT_CONFIG_PATH if os.path.exists(DEFAULT_CONFIG_PATH) else None)
    cfg = _sidebar(base)

    picked = _pick_image()
    if picked is None:
        st.info("Importez une image de sudoku pour commencer.")
        return
    name, data = picked
    img = decode_image(data)
    if img is None:
        st.error(f"{name} n'est pas une image lisible.")
        return
    image_id = hashlib.sha1(data).hexdigest()[:12]

    try:
        reading = _read(data, cfg["ocr"]["backend"], cfg["ocr"]["cnn_weights"],
                        cfg["predict"]["conf_min"], cfg["detect"]["warp_size"])
    except FileNotFoundError as e:
        st.error(f"{e}")
        st.code("python scripts/train_cnn.py --data data/assets", language="bash")
        return
    except RuntimeError as e:
        st.error(f"{e} Essayez une image plus nette, prise de face, où la grille est entière.")
        st.image(to_rgb(img), caption=name, width=400)
        return

    # --- correction : la grille éditée pilote tout le reste de la page
    reset = st.session_state.setdefault(f"reset-{image_id}", 0)
    st.subheader("Grille lue")
    col_edit, col_view = st.columns([5, 4], gap="large")
    with col_edit:
        st.caption("Cliquez une case pour corriger un chiffre ; videz-la pour une case vide.")
        edited = st.data_editor(
            grid_to_frame(reading.grid), key=f"editor-{image_id}-{reset}",
            column_config={c: st.column_config.NumberColumn(c, min_value=1, max_value=9, step=1,
                                                            width=44) for c in COLUMNS},
            num_rows="fixed", width="content", placeholder="",  # cases vides sans "None"
        )
        grid = frame_to_grid(edited)
        conflicts = find_conflicts(grid)
        n_solutions = 0 if conflicts else count_solutions(grid, limit=2)
        _status(grid, conflicts, n_solutions)
        n_changed = int(np.count_nonzero(grid != reading.grid))
        if n_changed and st.button(f"Annuler mes {n_changed} correction(s)"):
            st.session_state[f"reset-{image_id}"] += 1
            st.rerun()
    with col_view:
        tab_read, tab_detect = st.tabs(["Lecture", "Détection"])
        tab_read.image(to_rgb(draw_reading(reading.warped, reading.xs, reading.ys, grid, conflicts)),
                       caption="Grille redressée, lignes et chiffres retenus", width="stretch")
        tab_detect.image(to_rgb(draw_detection(img, reading.quad)),
                         caption="Contour détecté dans l'image d'origine", width="stretch")

    # --- solution
    st.subheader("Solution")
    if n_solutions != 1:
        st.info("La solution s'affiche dès que la grille est valide et à solution unique.")
        return
    solved = grid.copy()
    solve(solved)
    result = render_solution(img, reading, solved, grid != 0, cfg)
    # jamais agrandie au-delà de sa taille réelle (pixellisation)
    st.image(to_rgb(result), caption=f"{name} résolu", width=min(result.shape[1], 720))
    st.download_button("Télécharger l'image résolue", encode_png(result),
                       file_name=f"{Path(name).stem}_resolu.png", mime="image/png",
                       type="primary")


main()
