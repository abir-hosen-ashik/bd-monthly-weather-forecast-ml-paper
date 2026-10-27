import itertools
import random
import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from dl import TorchRegressor

warnings.filterwarnings("ignore")

BASELINES = ["Climatology", "Persistence", "Seasonal naive", "SARIMA"]
ML = ["Ridge", "DT", "RF", "SVR", "KNN", "XGB", "MLP", "LSTM", "GRU"]
STOCHASTIC = {"RF", "XGB", "MLP", "LSTM", "GRU"}
TORCH = {"MLP", "LSTM", "GRU"}
SEQUENCE = {"LSTM", "GRU"}

GRIDS = {
    "Ridge": {"alpha": [0.1, 1, 10, 100]},
    "DT": {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 5, 20, 50]},
    "RF": {"max_depth": [8, 16, None], "min_samples_leaf": [1, 5, 10], "max_features": [0.33, 0.6, 1.0]},
    "SVR": {"C": [1, 10, 100], "gamma": ["scale", 0.01], "epsilon": [0.05, 0.2]},
    "KNN": {"n_neighbors": [5, 10, 20, 40], "weights": ["uniform", "distance"], "p": [1, 2]},
    "XGB": {"n_estimators": [200, 500], "max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0], "colsample_bytree": [0.6, 1.0], "reg_lambda": [1, 10]},
    "MLP": {"hidden": [(64, 32), (128, 64)], "weight_decay": [1e-4, 1e-3], "dropout": [0.0, 0.2]},
    "LSTM": {"hidden": [32, 64]},
    "GRU": {"hidden": [32, 64]},
}
N_CONFIGS = {"Ridge": 4, "DT": 10, "RF": 8, "SVR": 6, "KNN": 8, "XGB": 10, "MLP": 4, "LSTM": 2, "GRU": 2}


def sample_configs(name, seed=0):
    grid = GRIDS[name]
    keys = list(grid)
    allc = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]
    k = min(N_CONFIGS[name], len(allc))
    return random.Random(seed).sample(allc, k) if k < len(allc) else allc


def make_estimator(name, params, seed=0):
    p = dict(params)
    if name == "Ridge":
        est = Ridge(**p)
    elif name == "DT":
        est = DecisionTreeRegressor(random_state=seed, **p)
    elif name == "RF":
        est = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=2, **p)
    elif name == "SVR":
        est = SVR(kernel="rbf", cache_size=1000, **p)
    elif name == "KNN":
        est = KNeighborsRegressor(**p)
    elif name == "XGB":
        est = XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=2, tree_method="hist", **p)
    elif name == "MLP":
        return TorchRegressor(kind="mlp", seed=seed, **p)
    elif name in ("LSTM", "GRU"):
        return TorchRegressor(kind=name.lower(), seed=seed, dropout=0.1, **p)
    else:
        raise ValueError(name)
    pipe = Pipeline([("scale", StandardScaler()), ("model", est)])
    if name in ("Ridge", "SVR", "KNN"):
        return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    return pipe


def fit_predict(name, params, seed, Xtr, ytr, Xte, Str=None, Ste=None, val_mask=None):
    est = make_estimator(name, params, seed)
    if name in TORCH:
        est.fit(Xtr, ytr, S=Str, val_mask=val_mask, refit=True)
        return est.predict(Xte, Ste), est
    est.fit(Xtr, ytr)
    return est.predict(Xte), est
