"""Interface web (Streamlit). Lancement : sudoku-ocr-ui [options streamlit]."""
import sys
from pathlib import Path


def main() -> int:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("Streamlit n'est pas installé : pip install -e '.[ui]'", file=sys.stderr)
        return 1
    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    return stcli.main()
