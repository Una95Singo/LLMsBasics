"""Plotting helpers shared across notebooks.

Only plotting glue lives here. Every model component (attention, layer norm,
transformer block, ...) is defined inline in the notebook that introduces it,
so nothing conceptually interesting is ever hidden behind an import.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _to_numpy(x):
    """Convert a torch tensor or array-like to a detached numpy array on CPU."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _printable(ch: str) -> str:
    """Render whitespace characters as visible glyphs for axis labels."""
    return {"\n": "\\n", "\t": "\\t", " ": "␣"}.get(ch, ch)


def plot_attention(
    attn,
    tokens: Sequence[str] | None = None,
    ax=None,
    title: str = "",
    cmap: str = "viridis",
    show_values: bool = False,
):
    """Heatmap of an attention matrix with token labels on both axes.

    attn: (T, T) torch tensor or numpy array. Row i = "token i's attention
          distribution over tokens 0..T-1".
    tokens: optional list of length T for axis labels.
    """
    A = _to_numpy(attn)
    T = A.shape[0]
    if ax is None:
        fig, ax = plt.subplots(figsize=(0.35 * T + 2, 0.35 * T + 1.5))
    im = ax.imshow(A, cmap=cmap, aspect="equal")
    if tokens is not None:
        labels = [_printable(t) for t in tokens]
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("key position (what we attend TO)")
    ax.set_ylabel("query position (who is attending)")
    if title:
        ax.set_title(title)
    if show_values and T <= 16:
        for i in range(T):
            for j in range(T):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                        color="w" if A[i, j] < 0.5 else "k", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_loss_curves(histories: dict, ax=None, title: str = "Loss", ylabel: str = "cross-entropy (nats)"):
    """Plot one or more loss curves on the same axes.

    histories: {label: list_or_array_of_values}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    for label, values in histories.items():
        v = np.asarray(values)
        ax.plot(v, label=label, linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_token_probs(probs, itos: dict, top_k: int = 10, ax=None, title: str = ""):
    """Horizontal bar chart of the top-k next-token probabilities."""
    p = _to_numpy(probs).reshape(-1)
    idx = np.argsort(p)[::-1][:top_k]
    labels = [_printable(itos[int(i)]) for i in idx]
    values = p[idx]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 0.35 * top_k + 1))
    y = np.arange(top_k)
    ax.barh(y, values, color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("probability")
    if title:
        ax.set_title(title)
    ax.set_xlim(0, max(values.max() * 1.15, 0.05))
    for i, v in enumerate(values):
        ax.text(v + values.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    return ax


def plot_embedding_pca(emb, itos: dict, ax=None, title: str = "Embedding PCA"):
    """Project a (vocab, d) embedding table to 2D with PCA and scatter with labels."""
    E = _to_numpy(emb)
    Ec = E - E.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    coords = Ec @ Vt.T[:, :2]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=8, color="#999", alpha=0.6)
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, _printable(itos[int(i)]), fontsize=9, ha="center", va="center")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def show_tokenization(text: str, stoi: dict, ax=None, width: int = 40):
    """Render characters as labeled colored boxes (one box per token-id)."""
    ids = [stoi[c] for c in text]
    rows = [text[i : i + width] for i in range(0, len(text), width)]
    id_rows = [ids[i : i + width] for i in range(0, len(ids), width)]
    nrows = len(rows)
    if ax is None:
        fig, ax = plt.subplots(figsize=(width * 0.25 + 1, nrows * 0.6 + 0.5))
    cmap = plt.get_cmap("tab20")
    for r, (chars, id_row) in enumerate(zip(rows, id_rows)):
        y = nrows - r - 1
        for c, (ch, tid) in enumerate(zip(chars, id_row)):
            color = cmap(tid % 20)
            ax.add_patch(plt.Rectangle((c, y), 1, 1, facecolor=color, alpha=0.5, edgecolor="w"))
            ax.text(c + 0.5, y + 0.65, _printable(ch), ha="center", va="center", fontsize=10)
            ax.text(c + 0.5, y + 0.25, str(tid), ha="center", va="center", fontsize=7, color="#333")
    ax.set_xlim(0, width)
    ax.set_ylim(0, nrows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plot_generation_grid(snapshots: Iterable[tuple], ax=None, title: str = "Samples during training"):
    """Render a table of (label, text) pairs. Useful for the Karpathy tour."""
    snaps = list(snapshots)
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 0.65 * len(snaps) + 0.6))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12)
    for i, (label, text) in enumerate(snaps):
        y = len(snaps) - i - 1
        ax.text(0.0, y + 0.5, str(label), fontsize=10, fontweight="bold",
                family="monospace", va="center")
        preview = text.replace("\n", " ⏎ ")
        if len(preview) > 140:
            preview = preview[:137] + "..."
        ax.text(0.17, y + 0.5, preview, fontsize=10, family="monospace", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(snaps))
    return ax


def style():
    """Apply a consistent, readable matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
