"""Post-LASSO regression + omega_GKW.

Fill in the four functions marked TODO.
Details, formulas, and grading are in README.md / PS_5.docx.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


NEWS_PREFIX = "news_"
CATEGORIES = ["Macro Release", "Central Bank", "Auction", "Ad Hoc"]


@dataclass
class PostLassoResult:
    selected_news: List[str]
    ols_results: sm.regression.linear_model.RegressionResultsWrapper
    coefficients: pd.DataFrame
    omega_gkw: float
    omega_gkw_by_category: Dict[str, float] = field(default_factory=dict)


def infer_category(var: str) -> str:
    """news_<category>_... -> category. Returns 'Other' if no match."""
    if not var.startswith(NEWS_PREFIX):
        return "Other"
    tail = var[len(NEWS_PREFIX):]
    for cat in CATEGORIES:
        if tail.startswith(cat + "_") or tail == cat:
            return cat
    return "Other"


def extract_coefficients(results) -> pd.DataFrame:
    """OLS results -> DataFrame with variable, coef, std_err, t_stat, p_value, abs_t."""
    coefs = pd.DataFrame({
        "variable": results.params.index,
        "coef": results.params.values,
        "std_err": results.bse.values,
        "t_stat": results.tvalues.values,
        "p_value": results.pvalues.values,
    })
    coefs["abs_t"] = coefs["t_stat"].abs()
    return coefs


# ---- TASK 2: K-fold LASSO on news_* columns  (see README §Task 2) ----------
def select_news_via_lasso(X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> List[str]:
    # TODO: implement -- return list of selected news_* column names
    news_cols = [c for c in X.columns if c.startswith(NEWS_PREFIX)]
    X_news = X[news_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_news_scaled = scaler.fit_transform(X_news)

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    lasso = LassoCV(cv=cv, max_iter=10000)
    lasso.fit(X_news_scaled, y.to_numpy(dtype=float))

    #buraya tekrardan bak!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # include this numbers in write up!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Task 3 diagnostics
    print(f"LASSO alpha_ (CV-picked penalty):        {lasso.alpha_:.6e}")
    print(f"LASSO alphas_[0] (approx alpha_max):     {lasso.alphas_[0]:.6e}")
    selected = [col for col, coef in zip(news_cols, lasso.coef_) if abs(coef) > 1e-10]
    print(f"Number of selected news variables:       {len(selected)}")
    assert lasso.alpha_ < lasso.alphas_[0], "CV picked alpha_max — zero variables selected!"

    return selected


# ---- TASK 4: post-LASSO OLS with HC1 SEs  (see README §Task 4) -------------
def fit_post_lasso_ols(X, y, selected_news: List[str], cov_type: str = "HC1"):
    # TODO: implement -- return statsmodels results wrapper
    control_cols = [c for c in X.columns if not c.startswith(NEWS_PREFIX)]
    final_cols = control_cols + selected_news
    X_sub = X[final_cols].astype(float)
    results = sm.OLS(y.astype(float), X_sub).fit(cov_type=cov_type)
    return results


# ---- TASK 5: overall omega_GKW  (see README §Task 5) -----------------------
def compute_omega_gkw(X: pd.DataFrame, y: pd.Series, coefficients: pd.DataFrame) -> float:
    # TODO: implement -- return float in [0, 1]
    news_cols_in_coefficients = [
        v for v in coefficients["variable"] if v.startswith(NEWS_PREFIX)
    ]
    any_active = X[news_cols_in_coefficients].to_numpy().any(axis=1)
    return float(y[any_active].sum() / y.sum())


# ---- TASK 6: per-category omega_GKW, winner-takes-all  (see README §Task 6) ----
def compute_omega_gkw_by_category(X, y, coefficients) -> Dict[str, float]:
    # TODO: implement -- return dict with all four CATEGORIES as keys
    news_coefs = (
        coefficients[coefficients["variable"].str.startswith(NEWS_PREFIX)]
        .sort_values("abs_t", ascending=False)
        .reset_index(drop=True)
    )

    # ranked list of news column names (highest |t-stat| first)
    ranked_cols = news_coefs["variable"].tolist()

    # numpy matrix: rows=bars, cols=ranked news dummies
    news_matrix = X[ranked_cols].to_numpy(dtype=bool)
    y_vals = y.to_numpy(dtype=float)
    total_y = y_vals.sum()

    category_sums: Dict[str, float] = {cat: 0.0 for cat in CATEGORIES}

    for t in range(len(y_vals)):
        # find first (highest-ranked) active dummy at this bar
        active_indices = np.where(news_matrix[t])[0]
        if active_indices.size == 0:
            continue
        winner_col = ranked_cols[active_indices[0]]
        cat = infer_category(winner_col)
        if cat in category_sums:
            category_sums[cat] += y_vals[t]

    return {cat: val / total_y for cat, val in category_sums.items()}


# --- Orchestrator (do not modify) -------------------------------------------
def run_post_lasso(X, y, cv_folds: int = 5, cov_type: str = "HC1") -> PostLassoResult:
    selected = select_news_via_lasso(X, y, cv_folds=cv_folds)
    ols = fit_post_lasso_ols(X, y, selected_news=selected, cov_type=cov_type)
    coefs = extract_coefficients(ols)
    return PostLassoResult(
        selected_news=selected,
        ols_results=ols,
        coefficients=coefs,
        omega_gkw=compute_omega_gkw(X, y, coefs),
        omega_gkw_by_category=compute_omega_gkw_by_category(X, y, coefs),
    )
