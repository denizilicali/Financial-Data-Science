"""Plot helpers for PS 6.

Fill in the two functions marked TODO.
Details and required elements are in README.md / PS_6.docx.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["OLS", "Huber", "ElasticNet", "PCR", "PLS", "RandomForest", "BoostedTrees"]


# ---- TASK 8a: bar chart of OOS R^2  (see README Task 8) --------------------
def plot_oos_r2_bar(
    oos_r2: Dict[str, float],
    save_path: Optional[Path] = None,
    title: str = "Out-of-sample R^2 by model",
) -> plt.Figure:
    names = [name for name in MODEL_ORDER if name in oos_r2]
    values = [oos_r2[name] for name in names]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, values, color=colors)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Out-of-sample R^2")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        offset = 0.001 if value >= 0 else -0.001
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.4f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


# ---- TASK 8b: heatmap of the Diebold-Mariano matrix  (see README Task 8) ---
def plot_dm_heatmap(
    dm_matrix: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "Diebold-Mariano statistics (row vs column)",
) -> plt.Figure:
    names = [name for name in MODEL_ORDER if name in dm_matrix.index]
    data = dm_matrix.loc[names, names].to_numpy(dtype=float)

    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_title(title)

    for i in range(len(names)):
        for j in range(len(names)):
            value = data[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="DM statistic")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig
