"""Download the Tiny Shakespeare corpus.

Saves ~1 MB of Shakespeare plays to data/tinyshakespeare.txt. Every notebook in
this repo reads from that file, so run this once before opening notebook 00.
"""
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = Path(__file__).parent / "tinyshakespeare.txt"


def main() -> None:
    if OUT.exists():
        print(f"Already have {OUT} ({OUT.stat().st_size:,} bytes). Skipping download.")
        return
    print(f"Downloading Tiny Shakespeare from {URL} ...")
    with urlopen(URL) as r:
        data = r.read()
    OUT.write_bytes(data)
    print(f"Saved {len(data):,} bytes to {OUT}")


if __name__ == "__main__":
    main()
