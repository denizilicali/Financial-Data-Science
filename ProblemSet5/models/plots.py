"""Plot helpers for PS 5.

Fill in the two functions marked TODO.
Details and required elements are in README.md / PS_5.docx.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from models.post_lasso_regression import CATEGORIES, NEWS_PREFIX


# ---- TASK 7: horizontal bar of top-N news by |t-stat|  (see README §Task 7) ----
def plot_top_news_by_tstat(
    coefficients: pd.DataFrame,
    n: int = 10,
    save_path: Optional[Path] = None,
    title: str = "Top news dummies by |t-stat| (post-LASSO OLS)",
) -> plt.Figure:
    # TODO: implement -- return the Figure, save to save_path if given
    news = (
        coefficients[coefficients["variable"].str.startswith(NEWS_PREFIX)]
        .sort_values("abs_t", ascending=False)
        .head(n)
        .sort_values("t_stat")
    )

    labels = news["variable"].str.removeprefix(NEWS_PREFIX).tolist()
    t_stats = news["t_stat"].tolist()
    colors = ["#e05c4b" if t < 0 else "#4c8cbf" for t in t_stats]

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.55)))
    ax.barh(labels, t_stats, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("t-stat")
    ax.set_title(title)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig


# ---- TASK 8: vertical bar of omega_GKW per category  (see README §Task 8) ----
def plot_omega_gkw_by_category(
    omega_by_category: Dict[str, float],
    save_path: Optional[Path] = None,
    title: str = "omega_GKW by news category",
) -> plt.Figure:
    # TODO: implement -- return the Figure, save to save_path if given
    cats = CATEGORIES
    values = [omega_by_category.get(c, 0.0) for c in cats]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(cats, values, color="#4c8cbf", edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("omega_GKW")
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    for bar, val in zip(bars, values):
        offset = 0.0005 if val >= 0 else -0.0010
        va = "bottom" if val >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:.4f}",
            ha="center",
            va=va,
            fontsize=9,
        )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig
