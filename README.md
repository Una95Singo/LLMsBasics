# LLMs from scratch — a bedtime reading series

A five-notebook journey that builds a small GPT from first principles, on a
laptop, using only PyTorch's tensor and autograd machinery. No
`nn.Transformer`, no `nn.MultiheadAttention` — every core mechanism is written
out so you can *see* it and, more importantly, see into it with plots.

The target reader is someone who already has a working understanding of neural
networks and gradient descent but wants intuition for **what a transformer is
and why it was the innovation behind systems like Claude and GPT.**

Each notebook is meant to be read in 15–25 minutes. One per night is a
reasonable pace.

## Table of contents

| # | Notebook | What you'll leave with |
|---|----------|------------------------|
| 00 | [`00_setup_and_big_picture.ipynb`](notebooks/00_setup_and_big_picture.ipynb) | A one-paragraph definition of an LLM, a tokenized corpus, and a map of the journey. |
| 01 | [`01_bigram_language_models.ipynb`](notebooks/01_bigram_language_models.ipynb) | The simplest possible language model, counted *and* learned. The connection between "counting" and "gradient descent." |
| 02 | [`02_rnns_and_their_limits.ipynb`](notebooks/02_rnns_and_their_limits.ipynb) | A tiny RNN built by hand; the sequential bottleneck and vanishing gradients shown, not just described. Motivation for what comes next. |
| 03 | [`03_attention_the_key_idea.ipynb`](notebooks/03_attention_the_key_idea.ipynb) | Self-attention from scratch, multi-head, positional encodings. The centerpiece. Attention heatmaps that visibly form structure during training. |
| 04 | [`04_the_transformer_block.ipynb`](notebooks/04_the_transformer_block.ipynb) | Residuals, LayerNorm, MLPs. The full block, stacked, with shape traces and parameter budgets. |
| 05 | [`05_training_minigpt.ipynb`](notebooks/05_training_minigpt.ipynb) | Train a 0.2 M-parameter mini-GPT on Shakespeare in minutes on CPU. Watch samples evolve from noise into near-English. A map from "this little model" to "Claude." |

## Setup

```bash
pip install -r requirements.txt
python data/download_shakespeare.py
jupyter lab
```

Then open the `notebooks/` folder and read 00 through 05 in order.

Each notebook is largely self-contained (it re-does the minimum setup it
needs), so you can also jump back into, say, notebook 03 on a second reading
without running the earlier ones.

## What's inside the code

- **`src/viz.py`** — plotting helpers only (attention heatmaps, loss curves,
  token-probability bars, embedding PCA, tokenization visualization). Nothing
  model-related is hidden here; you'll always see the tensor ops in the
  notebook that's teaching that concept.
- **`data/download_shakespeare.py`** — fetches the ~1 MB Tiny Shakespeare
  corpus from Karpathy's mirror.
- **`notebooks/`** — the five notebooks. Each contains its own minimal,
  readable implementation of whatever it's teaching.

## Inspiration and further reading

The spiritual parent of this repo is Andrej Karpathy's
[makemore](https://github.com/karpathy/makemore) and
[nanoGPT](https://github.com/karpathy/nanoGPT) — and his
["Zero to Hero" lecture series](https://karpathy.ai/zero-to-hero.html), which
you should absolutely watch. The difference in emphasis here: fewer features,
smaller models, more pictures, explicitly structured as bedtime reading.

For the original transformer paper, read [*Attention Is All You Need* (Vaswani
et al., 2017)](https://arxiv.org/abs/1706.03762).
