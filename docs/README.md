# LLMs from scratch — readable copies

These are Markdown exports of the notebooks in [`../notebooks/`](../notebooks).
They exist because the **GitHub iOS app doesn't render `.ipynb`** — only
`.md` — and the whole point of this series is being readable on a phone before
bed.

Every plot and printed output is preserved as an image in the matching
`<name>_files/` folder next to the `.md`.

## Read in order

1. [00 — Setup and the big picture](00_setup_and_big_picture.md)
2. [01 — Bigram language models](01_bigram_language_models.md)
3. [02 — RNNs and why they hit a wall](02_rnns_and_their_limits.md)
4. *03 — Attention: the key idea* (coming)
5. *04 — The transformer block* (coming)
6. *05 — Training a mini-GPT* (coming)

## Regenerating these files

After editing or re-executing any notebook, re-run:

```bash
python build/export_markdown.py
```

from the repo root.
