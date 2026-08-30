"""ML asset-pricing pipeline for PS 6.

Fill in the eight functions marked TODO. Details, formulas, hyperparameter
grids, and grading are in README.md / PS_6.docx.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


TARGET = "excess_return"
FEATURE_PREFIX = "z"
MODEL_NAMES = ["OLS", "Huber", "ElasticNet", "PCR", "PLS", "RandomForest", "BoostedTrees"]
ELASTIC_ALPHAS = [0.01, 0.1, 1.0, 10.0]
ELASTIC_L1_RATIOS = [0.1, 0.5, 0.9]
MAX_COMPONENTS = 15
RANDOM_STATE = 42


@dataclass
class Splits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


@dataclass
class MLResult:
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    oos_r2: Dict[str, float] = field(default_factory=dict)
    dm_matrix: pd.DataFrame = None
    best_elastic: Tuple[float, float] = (float("nan"), float("nan"))
    best_pcr_components: int = 0
    best_pls_components: int = 0


# ---- TASK 2: train/val/test split  (see README Task 2) ---------------------
def make_splits(X: np.ndarray, y: np.ndarray, test_size: float = 0.3,
                val_size: float = 0.2857, random_state: int = RANDOM_STATE) -> Splits:
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size, random_state=random_state
    )
    return Splits(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
    )


# ---- TASK 3: OLS + Huber baselines  (see README Task 3) --------------------
def fit_linear_baselines(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    ols_model = LinearRegression().fit(X_train, y_train)
    huber_model = HuberRegressor(max_iter=200).fit(X_train, y_train)
    return {"OLS": ols_model, "Huber": huber_model}


# ---- TASK 4: ElasticNet grid search on the validation set  (Task 4) --------
def fit_elasticnet_cv(X_train, y_train, X_val, y_val,
                      alphas: List[float] = ELASTIC_ALPHAS,
                      l1_ratios: List[float] = ELASTIC_L1_RATIOS):
    best_alpha, best_l1_ratio, best_mse = None, None, np.inf
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            model = ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
                max_iter=1000, random_state=RANDOM_STATE,
            ).fit(X_train, y_train)
            mse = mean_squared_error(y_val, model.predict(X_val))
            if mse < best_mse:
                best_mse = mse
                best_alpha, best_l1_ratio = alpha, l1_ratio

    final_model = ElasticNet(
        alpha=best_alpha, l1_ratio=best_l1_ratio, fit_intercept=True,
        max_iter=1000, random_state=RANDOM_STATE,
    ).fit(X_train, y_train)
    return final_model, best_alpha, best_l1_ratio


# ---- TASK 5: PCR + PLS with validation-set n_components search  (Task 5) ---
def fit_pcr_pls_cv(X_train, y_train, X_val, y_val,
                   max_components: int = MAX_COMPONENTS):
    n_components_range = range(1, min(X_train.shape[1], max_components) + 1)

    # ---- PCR ----
    best_pcr_k, best_pcr_mse = None, np.inf
    for k in n_components_range:
        pca = PCA(n_components=k).fit(X_train)
        X_train_pc = pca.transform(X_train)
        X_val_pc = pca.transform(X_val)
        lr = LinearRegression().fit(X_train_pc, y_train)
        mse = mean_squared_error(y_val, lr.predict(X_val_pc))
        if mse < best_pcr_mse:
            best_pcr_mse = mse
            best_pcr_k = k

    pca_final = PCA(n_components=best_pcr_k).fit(X_train)
    lr_final = LinearRegression().fit(pca_final.transform(X_train), y_train)

    # ---- PLS ----
    best_pls_k, best_pls_mse = None, np.inf
    for k in n_components_range:
        pls = PLSRegression(n_components=k).fit(X_train, y_train)
        mse = mean_squared_error(y_val, pls.predict(X_val).flatten())
        if mse < best_pls_mse:
            best_pls_mse = mse
            best_pls_k = k

    pls_final = PLSRegression(n_components=best_pls_k).fit(X_train, y_train)

    return (
        {"PCR": (pca_final, lr_final), "PLS": pls_final},
        best_pcr_k,
        best_pls_k,
    )


# ---- TASK 6: tree ensembles  (see README Task 6) ---------------------------
def fit_tree_ensembles(X_train, y_train, random_state: int = RANDOM_STATE) -> Dict[str, object]:
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=5, min_samples_split=5, random_state=random_state,
    ).fit(X_train, y_train)
    brt = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=random_state,
    ).fit(X_train, y_train)
    return {"RandomForest": rf, "BoostedTrees": brt}


# ---- TASK 7a: out-of-sample R^2  (see README Task 7) -----------------------
def compute_oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


# ---- TASK 7b: Newey-West standard error (helper for DM) --------------------
def newey_west_se(d: np.ndarray, lag: int = 1) -> float:
    n = len(d)
    dbar = np.mean(d)
    demeaned = d - dbar

    gamma_0 = np.sum(demeaned * demeaned) / n
    sigma2 = gamma_0
    for i in range(1, lag + 1):
        gamma_i = np.sum(demeaned[: n - i] * demeaned[i:]) / n
        sigma2 += 2 * (1 - i / (lag + 1)) * gamma_i

    return np.sqrt(sigma2 / n)


# ---- TASK 7c: Diebold-Mariano matrix  (see README Task 7) ------------------
def compute_dm_matrix(errors: Dict[str, np.ndarray], lag: int = 1) -> pd.DataFrame:
    names = list(errors.keys())
    dm = pd.DataFrame(0.0, index=names, columns=names)

    for i, name_i in enumerate(names):
        for name_j in names[i + 1:]:
            e_i = errors[name_i]
            e_j = errors[name_j]
            d_ij = e_i ** 2 - e_j ** 2
            se = newey_west_se(d_ij, lag=lag)
            dm_ij = np.mean(d_ij) / se
            dm.loc[name_i, name_j] = dm_ij
            dm.loc[name_j, name_i] = -dm_ij

    return dm


# ---- Orchestrator (do not modify) ------------------------------------------
def run_ml_pipeline(X: np.ndarray, y: np.ndarray) -> MLResult:
    splits = make_splits(X, y)

    linear = fit_linear_baselines(splits.X_train, splits.y_train)
    enet, best_alpha, best_l1 = fit_elasticnet_cv(
        splits.X_train, splits.y_train, splits.X_val, splits.y_val
    )
    dim_models, pcr_k, pls_k = fit_pcr_pls_cv(
        splits.X_train, splits.y_train, splits.X_val, splits.y_val
    )
    trees = fit_tree_ensembles(splits.X_train, splits.y_train)

    pca_pcr, lr_pcr = dim_models["PCR"]
    pls_model = dim_models["PLS"]

    preds = {
        "OLS": linear["OLS"].predict(splits.X_test),
        "Huber": linear["Huber"].predict(splits.X_test),
        "ElasticNet": enet.predict(splits.X_test),
        "PCR": lr_pcr.predict(pca_pcr.transform(splits.X_test)),
        "PLS": pls_model.predict(splits.X_test).flatten(),
        "RandomForest": trees["RandomForest"].predict(splits.X_test),
        "BoostedTrees": trees["BoostedTrees"].predict(splits.X_test),
    }

    r2 = {name: compute_oos_r2(splits.y_test, p) for name, p in preds.items()}
    errors = {name: splits.y_test - p for name, p in preds.items()}
    dm = compute_dm_matrix(errors, lag=1)

    return MLResult(
        predictions=preds,
        oos_r2=r2,
        dm_matrix=dm,
        best_elastic=(best_alpha, best_l1),
        best_pcr_components=pcr_k,
        best_pls_components=pls_k,
    )
