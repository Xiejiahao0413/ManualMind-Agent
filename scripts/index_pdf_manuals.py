import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pdf_manuals import PDF_MANUALS_DIR, index_pdf_manuals


def main() -> None:
    summary = index_pdf_manuals(PDF_MANUALS_DIR)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
