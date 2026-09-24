import argparse
import os
import sys

from .config import DEFAULT_CONFIG_PATH, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sudoku-ocr",
        description="Détecte, lit et résout une grille de sudoku dans une image.",
    )
    parser.add_argument('--image', required=True, help="image d'entrée")
    parser.add_argument('--out', default='data/outputs/result.jpg', help="image de sortie")
    parser.add_argument('--config', default=None,
                        help=f"fichier YAML appliqué par-dessus {DEFAULT_CONFIG_PATH} "
                             f"(qui est toujours lu en premier s'il existe)")
    parser.add_argument('--backend', choices=['cnn', 'tesseract'], default=None,
                        help="surcharge ocr.backend")
    parser.add_argument('--weights', default=None, help="surcharge ocr.cnn_weights")
    return parser


def config_from_args(args: argparse.Namespace) -> dict:
    # default.yaml d'abord, puis --config ne remplace que les clés qu'il contient
    paths = []
    if os.path.exists(DEFAULT_CONFIG_PATH):
        paths.append(DEFAULT_CONFIG_PATH)
    if args.config is not None:
        paths.append(args.config)
    ocr = {}
    if args.backend is not None:
        ocr["backend"] = args.backend
    if args.weights is not None:
        ocr["cnn_weights"] = args.weights
    return load_config(paths, {"ocr": ocr} if ocr else None)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))

    from .pipeline import run
    try:
        run(args.image, args.out, cfg)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
