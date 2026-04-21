"""Re-export all notebooks in notebooks/ to docs/ as Markdown.

This produces GitHub-iOS-app-friendly copies of each notebook. Run this any
time a notebook changes:

    python build/export_markdown.py
"""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "docs"
OUT_DIR.mkdir(exist_ok=True)

notebooks = sorted(NB_DIR.glob("*.ipynb"))
if not notebooks:
    raise SystemExit(f"No notebooks found in {NB_DIR}")

for nb in notebooks:
    print(f"Exporting {nb.name}")
    subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "markdown",
            "--output-dir", str(OUT_DIR),
            str(nb),
        ],
        check=True,
    )

print(f"\nDone. Markdown copies written to {OUT_DIR}")
