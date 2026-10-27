import json
import importlib

import numpy as np
import pandas as pd

from common import RESULTS, TARGETS, metrics
from features import DYN, add_fold_features, add_neighbour_lag, tabular_features
from models import make_estimator

pf = importlib.import_module("10_panel_forecast")
OUT = RESULTS / "spatial"
OUT.mkdir(exist_ok=True, parents=True)
MODELS = ["Ridge", "RF", "XGB"]
SEEDS = [0, 1, 2]


def main():
    log = json.loads((RESULTS / "panel" / "tuning.json").read_text())
    params = {(r["Target"], r["Model"]): r["best"] for r in log if "best" in r}
    raw, df0 = pf.load_panel()
    rows = []
    for target in TARGETS:
        feats = tabular_features(target, DYN)
        pooled = add_fold_features(df0, target, df0.Year < 2018)
        for div in sorted(df0.Division.unique()):
            train_st = sorted(df0.loc[df0.Division != div, "Station"].unique())
            dl = add_neighbour_lag(df0, target, train_st)
            dl = add_fold_features(dl, target, dl.Year < 2018)
            tr = dl[(dl.Year < 2018) & (dl.Division != div)]
            te = dl[(dl.Year >= 2018) & (dl.Division == div)]
            ptr = pooled[pooled.Year < 2018]
            pte = pooled[(pooled.Year >= 2018) & (pooled.Division == div)]
            rows.append({"Target": target, "Division": div, "Setting": "Climatology", "Model": "Climatology", "Seed": 0,
                         "n_test_stations": te.Station.nunique(), **metrics(te[target], te.clim, te.clim)})
            for mdl in MODELS:
                for sd in (SEEDS if mdl != "Ridge" else [0]):
                    for setting, a, b in [("LODO", tr, te), ("Pooled", ptr, pte)]:
                        est = make_estimator(mdl, params[(target, mdl)], sd).fit(a[feats].to_numpy(float), a[target].to_numpy(float))
                        p = est.predict(b[feats].to_numpy(float))
                        if target == "Rainfall":
                            p = np.clip(p, 0, None)
                        rows.append({"Target": target, "Division": div, "Setting": setting, "Model": mdl, "Seed": sd,
                                     "n_test_stations": b.Station.nunique(), **metrics(b[target], p, b.clim)})
            print(target, div, "done", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "lodo.csv", index=False)


if __name__ == "__main__":
    main()
