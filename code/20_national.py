import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from common import DATA, INNER_VAL_BLOCKS, RESULTS, SEEDS, SRC_NATIONAL_1980, SRC_NATIONAL_1999, metrics, r2

warnings.filterwarnings("ignore")
OUT = RESULTS / "national"
OUT.mkdir(exist_ok=True, parents=True)

COVARS = {"Rainfall": ["Temperature", "Humidity"], "Temperature": ["Rainfall", "Humidity"],
          "Humidity": ["Temperature", "Rainfall"]}
MODELS = ["DT", "RF", "SVR", "KNN", "MLP", "XGB"]
MANUAL = {
    "Rainfall": {"DT": dict(max_depth=5), "RF": dict(n_estimators=200, max_depth=10),
                 "SVR": dict(C=100, gamma=0.1, epsilon=0.1), "KNN": dict(n_neighbors=5),
                 "MLP": dict(hidden_layer_sizes=(100, 50)), "XGB": dict(n_estimators=100, learning_rate=0.1)},
    "Temperature": {"DT": dict(), "RF": dict(n_estimators=200, max_depth=10),
                    "SVR": dict(C=100, gamma=0.1, epsilon=0.1), "KNN": dict(n_neighbors=5),
                    "MLP": dict(hidden_layer_sizes=(100, 50)),
                    "XGB": dict(n_estimators=100, learning_rate=0.1, max_depth=5, colsample_bytree=0.3, reg_alpha=10)},
}
MANUAL["Humidity"] = MANUAL["Rainfall"]
GRID = {
    "DT": {"max_depth": [2, 3, 4, 5, 7, 10, None], "min_samples_leaf": [1, 3, 5, 10, 20]},
    "RF": {"n_estimators": [200], "max_depth": [4, 8, None], "min_samples_leaf": [1, 5],
           "max_features": [0.5, 1.0]},
    "SVR": {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, "scale", 1.0], "epsilon": [0.01, 0.1]},
    "KNN": {"n_neighbors": [3, 5, 9, 15, 25], "weights": ["uniform", "distance"]},
    "MLP": {"hidden_layer_sizes": [(32,), (100, 50)], "alpha": [1e-3, 1e-1, 1.0]},
    "XGB": {"n_estimators": [100, 300], "max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1]},
}


def base_est(name, p, seed=0):
    p = dict(p)
    if name == "DT":
        return DecisionTreeRegressor(random_state=seed, **p)
    if name == "RF":
        p.setdefault("n_estimators", 200)
        return RandomForestRegressor(random_state=seed, n_jobs=2, **p)
    if name == "SVR":
        return SVR(kernel="rbf", **p)
    if name == "KNN":
        return KNeighborsRegressor(**p)
    if name == "MLP":
        return MLPRegressor(max_iter=1000, random_state=seed, early_stopping=False, **p)
    if name == "XGB":
        return XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=2, **p)


def corrected_est(name, p, seed=0):
    pipe = Pipeline([("scale", StandardScaler()), ("model", base_est(name, p, seed))])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def national_from_panel(path, years=None):
    p = pd.read_csv(path)
    n = p.groupby(["Year", "Month"], as_index=False)[["Temperature", "Rainfall", "Humidity"]].mean()
    return n if years is None else n[n.Year.between(*years)]


def original_features(df, target):
    d = df.copy()
    d["Month_Sin"] = np.sin(2 * np.pi * d["Month"] / 12)
    d["Month_Cos"] = np.cos(2 * np.pi * d["Month"] / 12)
    d["Year_Sin"] = np.sin(2 * np.pi * d["Year"] / d["Year"].max())
    d["Year_Cos"] = np.cos(2 * np.pi * d["Year"] / d["Year"].max())
    X = d[["Month_Sin", "Month_Cos", "Year_Sin", "Year_Cos"] + COVARS[target]].to_numpy(float)
    return X


def corrected_features(df, target):
    d = df.copy()
    X = np.column_stack([np.sin(2 * np.pi * d["Month"] / 12), np.cos(2 * np.pi * d["Month"] / 12),
                         d["Year"] - 2000] + [d[c] for c in COVARS[target]])
    return X


def run_P0(df, target, name):
    y = df[target].to_numpy(float)
    lo, hi = y.min(), y.max()
    yn = (y - lo) / (hi - lo)
    X = original_features(df, target)
    tr, te = (df.Year < 2018).to_numpy(), (df.Year >= 2018).to_numpy()
    est = base_est(name, MANUAL[target][name], 42)
    est.fit(X[tr], yn[tr])
    return est.predict(X[te]) * (hi - lo) + lo, y[te], est.predict(X[tr]) * (hi - lo) + lo, y[tr]


def inner_score(df, target, name, p, featfun=corrected_features, scaled=True):
    sq = []
    for a, b in INNER_VAL_BLOCKS:
        tr = (df.Year < a).to_numpy()
        va = ((df.Year >= a) & (df.Year <= b)).to_numpy()
        X, y = featfun(df, target), df[target].to_numpy(float)
        est = corrected_est(name, p) if scaled else base_est(name, p)
        est.fit(X[tr], y[tr])
        sq.append((est.predict(X[va]) - y[va]) ** 2)
    return float(np.mean(np.concatenate(sq)))


def configs(name):
    g = GRID[name]
    return [dict(zip(g, v)) for v in itertools.product(*g.values())]


def run_corrected(df, target, name, tuned, seeds):
    X, y = corrected_features(df, target), df[target].to_numpy(float)
    tr, te = (df.Year < 2018).to_numpy(), (df.Year >= 2018).to_numpy()
    if tuned:
        dev = df[df.Year <= 2017].reset_index(drop=True)
        scores = [(inner_score(dev, target, name, c), c) for c in configs(name)]
        best = min(scores, key=lambda t: t[0])[1]
    else:
        best = MANUAL[target][name]
    out = []
    for sd in seeds:
        est = corrected_est(name, best, sd)
        est.fit(X[tr], y[tr])
        out.append((est.predict(X[te]), est.predict(X[tr])))
    return best, out, y[te], y[tr]


def main():
    sets = {
        "1980-2022 (original series)": pd.read_csv(SRC_NATIONAL_1980),
        "1999-2022 (original series)": pd.read_csv(SRC_NATIONAL_1999),
        "1981-2022 (QC series, 24 st.)": national_from_panel(DATA / "panel_monthly_24st.csv"),
        "1999-2022 (QC series, 33 st.)": national_from_panel(DATA / "panel_monthly_33st.csv"),
    }
    rows, best_params = [], []
    for sname, df in sets.items():
        df = df.sort_values(["Year", "Month"]).reset_index(drop=True)
        for target in ["Rainfall", "Temperature", "Humidity"]:
            clim_te = df[df.Year < 2018].groupby("Month")[target].mean().reindex(df[df.Year >= 2018].Month).to_numpy()
            for name in MODELS:
                stoch = name in ("RF", "MLP", "XGB")
                if "original" in sname:
                    p, yte, ptr, ytr = run_P0(df, target, name)
                    rows.append({"Series": sname, "Protocol": "P0 original", "Target": target, "Model": name, "Seed": 42,
                                 **{f"test_{k}": v for k, v in metrics(yte, p, clim_te).items()},
                                 **{f"train_{k}": v for k, v in metrics(ytr, ptr).items()}})
                    best, outs, yte, ytr = run_corrected(df, target, name, False, SEEDS if stoch else [0])
                    for sd, (p, ptr) in zip(SEEDS, outs):
                        rows.append({"Series": sname, "Protocol": "P1 scaled, manual", "Target": target, "Model": name,
                                     "Seed": sd, **{f"test_{k}": v for k, v in metrics(yte, p, clim_te).items()},
                                     **{f"train_{k}": v for k, v in metrics(ytr, ptr).items()}})
                proto = "P2 scaled, tuned" if "original" in sname else "P3 QC series, scaled, tuned"
                best, outs, yte, ytr = run_corrected(df, target, name, True, SEEDS if stoch else [0])
                best_params.append({"Series": sname, "Target": target, "Model": name, "params": str(best)})
                for sd, (p, ptr) in zip(SEEDS, outs):
                    rows.append({"Series": sname, "Protocol": proto, "Target": target, "Model": name, "Seed": sd,
                                 **{f"test_{k}": v for k, v in metrics(yte, p, clim_te).items()},
                                 **{f"train_{k}": v for k, v in metrics(ytr, ptr).items()}})
                print(sname, target, name, best, flush=True)
            rows.append({"Series": sname, "Protocol": "reference", "Target": target, "Model": "Climatology", "Seed": 0,
                         **{f"test_{k}": v for k, v in metrics(df[df.Year >= 2018][target], clim_te, clim_te).items()}})
    pd.DataFrame(rows).to_csv(OUT / "national_protocols.csv", index=False)
    pd.DataFrame(best_params).to_csv(OUT / "national_best_params.csv", index=False)
    sensitivity(sets["1980-2022 (original series)"].sort_values(["Year", "Month"]).reset_index(drop=True))


def sensitivity(df):
    tr, te = (df.Year < 2018).to_numpy(), (df.Year >= 2018).to_numpy()
    rows = []
    for target in ["Temperature", "Rainfall"]:
        X, y = corrected_features(df, target), df[target].to_numpy(float)
        for name in ["SVR", "KNN", "MLP"]:
            for scaling in ["none", "minmax", "standard"]:
                for sd in (SEEDS if name == "MLP" else [0]):
                    est = base_est(name, MANUAL[target][name], sd)
                    if scaling != "none":
                        sc = MinMaxScaler() if scaling == "minmax" else StandardScaler()
                        est = TransformedTargetRegressor(Pipeline([("s", sc), ("m", est)]), transformer=StandardScaler())
                    else:
                        est = TransformedTargetRegressor(est, transformer=MinMaxScaler())
                    est.fit(X[tr], y[tr])
                    rows.append({"analysis": "scaling", "Target": target, "Model": name, "scaling": scaling, "Seed": sd,
                                 "test_R2": r2(y[te], est.predict(X[te]))})
    dev = df[df.Year <= 2017].reset_index(drop=True)
    for target in ["Temperature", "Rainfall"]:
        X, y = corrected_features(df, target), df[target].to_numpy(float)
        var_dev = np.var(dev[target])
        for C, gm in itertools.product([0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10]):
            p = dict(C=C, gamma=gm, epsilon=0.1)
            est = corrected_est("SVR", p).fit(X[tr], y[tr])
            rows.append({"analysis": "svr_surface", "Target": target, "C": C, "gamma": gm,
                         "inner_MSE": inner_score(dev, target, "SVR", p), "test_R2": r2(y[te], est.predict(X[te]))})
        for h, al in itertools.product([(16,), (32,), (100, 50), (100, 100)], [1e-4, 1e-2, 1e-1, 1.0]):
            for sd in SEEDS[:3]:
                est = corrected_est("MLP", dict(hidden_layer_sizes=h, alpha=al), sd).fit(X[tr], y[tr])
                rows.append({"analysis": "mlp_surface", "Target": target, "hidden": str(h), "alpha": al, "Seed": sd,
                             "test_R2": r2(y[te], est.predict(X[te]))})
    pd.DataFrame(rows).to_csv(OUT / "national_sensitivity.csv", index=False)
    mlp_curves(df)


def mlp_curves(df, epochs=1500):
    rows = []
    for target in ["Temperature", "Rainfall"]:
        y = df[target].to_numpy(float)
        tr = (df.Year < 2014).to_numpy()
        va = ((df.Year >= 2014) & (df.Year <= 2017)).to_numpy()
        te = (df.Year >= 2018).to_numpy()
        for variant in ["original (unscaled X)", "standardised X"]:
            X = original_features(df, target) if variant.startswith("original") else corrected_features(df, target)
            ys = MinMaxScaler().fit(y[tr].reshape(-1, 1))
            if variant.startswith("standardised"):
                sx = StandardScaler().fit(X[tr])
                X = sx.transform(X)
            yt = ys.transform(y.reshape(-1, 1)).ravel()
            for sd in SEEDS[:5]:
                m = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=sd, learning_rate_init=1e-3)
                for ep in range(1, epochs + 1):
                    m.partial_fit(X[tr], yt[tr])
                    if ep % 5 and ep != 1:
                        continue
                    rec = {"Target": target, "variant": variant, "Seed": sd, "epoch": ep}
                    for nm, mk in [("train", tr), ("val", va), ("test", te)]:
                        p = ys.inverse_transform(m.predict(X[mk]).reshape(-1, 1)).ravel()
                        rec[f"{nm}_loss"] = float(np.mean((m.predict(X[mk]) - yt[mk]) ** 2))
                        rec[f"{nm}_R2"] = r2(y[mk], p)
                    rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "national_mlp_curves.csv", index=False)


if __name__ == "__main__":
    main()
