import json

import numpy as np
import pandas as pd
import shap
import torch

from common import RESULTS, TARGETS
from dl import TorchRegressor
from features import DYN, add_fold_features, sequence_arrays, static_features, tabular_features
from models import make_estimator

import importlib
pf = importlib.import_module("10_panel_forecast")

OUT = RESULTS / "diagnostics"
OUT.mkdir(exist_ok=True, parents=True)
N_SHAP = 1440


def tuned_params():
    log = json.loads((RESULTS / "panel" / "tuning.json").read_text())
    return {(r["Target"], r["Model"]): r["best"] for r in log if "best" in r}


def main():
    params = tuned_params()
    raw, df = pf.load_panel()
    imp_rows, inter_rows = [], []
    for target in TARGETS:
        trm = df.Year <= 2017
        d = add_fold_features(df, target, trm)
        dtr, dte = d[trm], d[d.Year >= 2018]
        feats = tabular_features(target, DYN)
        Xtr, Xte = dtr[feats].to_numpy(float), dte[feats].to_numpy(float)
        sample = dte.sample(min(N_SHAP, len(dte)), random_state=0)
        Xs = sample[feats].to_numpy(float)
        for mdl in ["RF", "XGB"]:
            est = make_estimator(mdl, params[(target, mdl)], 0).fit(Xtr, dtr[target].to_numpy(float))
            scaler, model = est.named_steps["scale"], est.named_steps["model"]
            Z = scaler.transform(Xs)
            ex = shap.TreeExplainer(model)
            sv = ex.shap_values(Z)
            np.save(OUT / f"shap_{target}_{mdl}.npy", sv)
            sample[feats].to_csv(OUT / f"shap_X_{target}_{mdl}.csv", index=False)
            for f, v in zip(feats, np.abs(sv).mean(0)):
                imp_rows.append({"Target": target, "Model": mdl, "feature": f, "mean_abs_shap": v})
            if mdl == "XGB":
                iv = ex.shap_interaction_values(Z[:500])
                m = np.abs(iv).mean(0)
                np.fill_diagonal(m, 0)
                for i in range(len(feats)):
                    for j in range(i + 1, len(feats)):
                        inter_rows.append({"Target": target, "f1": feats[i], "f2": feats[j], "mean_abs_interaction": 2 * m[i, j]})
            print(target, mdl, "SHAP done", flush=True)
    pd.DataFrame(imp_rows).to_csv(OUT / "shap_importance.csv", index=False)
    pd.DataFrame(inter_rows).sort_values("mean_abs_interaction", ascending=False).to_csv(OUT / "shap_interactions.csv", index=False)
    learning_curves(df, params)


def learning_curves(df, params):
    rows = []
    for target in TARGETS:
        trm = df.Year <= 2017
        d = add_fold_features(df, target, trm)
        dtr, dte = d[trm], d[d.Year >= 2018]
        vm = (dtr.Year >= 2015).to_numpy()
        for mdl in ["MLP", "LSTM", "GRU"]:
            p = dict(params[(target, mdl)])
            dropout = p.pop("dropout", 0.1)
            if mdl == "MLP":
                Xtr, Xte = dtr[tabular_features(target, DYN)].to_numpy(float), dte[tabular_features(target, DYN)].to_numpy(float)
                Str = Ste = None
            else:
                Xtr, Xte = sequence_arrays(dtr, DYN), sequence_arrays(dte, DYN)
                Str, Ste = dtr[static_features(target)].to_numpy(float), dte[static_features(target)].to_numpy(float)
            ytr, yte = dtr[target].to_numpy(float), dte[target].to_numpy(float)
            for sd in range(2):
                r = TorchRegressor(kind=mdl.lower(), seed=sd, curves=True, dropout=dropout, **p)
                r._scale_fit(Xtr[~vm], None if Str is None else Str[~vm], ytr[~vm])
                hist = []
                model = r._build(Xtr.shape[-1], 0 if Str is None else Str.shape[1])
                opt = torch.optim.Adam(model.parameters(), lr=r.lr, weight_decay=r.weight_decay)
                lossf = torch.nn.MSELoss()
                Xa, Sa = r._tx(Xtr[~vm], None if Str is None else Str[~vm])
                Xv, Sv = r._tx(Xtr[vm], None if Str is None else Str[vm])
                Xt, St = r._tx(Xte, Ste)
                ya = torch.tensor((ytr[~vm] - r.ym) / r.ys, dtype=torch.float32)
                g = torch.Generator().manual_seed(sd)
                for ep in range(1, 151):
                    model.train()
                    perm = torch.randperm(len(ya), generator=g)
                    for i in range(0, len(ya), r.batch_size):
                        idx = perm[i:i + r.batch_size]
                        opt.zero_grad()
                        loss = lossf(model(Xa[idx], None if Sa is None else Sa[idx]), ya[idx])
                        loss.backward()
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        rec = {"Target": target, "Model": mdl, "Seed": sd, "epoch": ep}
                        for nm, X_, S_, y_ in [("train", Xa, Sa, ytr[~vm]), ("val", Xv, Sv, ytr[vm]), ("test", Xt, St, yte)]:
                            pr = model(X_, S_).numpy() * r.ys + r.ym
                            rec[f"{nm}_loss"] = float(np.mean(((pr - y_) / r.ys) ** 2))
                            rec[f"{nm}_R2"] = float(1 - np.sum((y_ - pr) ** 2) / np.sum((y_ - y_.mean()) ** 2))
                        rows.append(rec)
                print(target, mdl, sd, "curve done", flush=True)
                pd.DataFrame(rows).to_csv(OUT / "dl_learning_curves.csv", index=False)


if __name__ == "__main__":
    main()
