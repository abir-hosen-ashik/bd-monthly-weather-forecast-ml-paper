import json
import importlib

import numpy as np
import pandas as pd

from common import DATA, RESULTS, TARGETS, metrics
from features import add_fold_features, tabular_features

pf = importlib.import_module("10_panel_forecast")
from models import make_estimator

OUT = RESULTS / "intervals"
OUT.mkdir(exist_ok=True, parents=True)
DYN3 = ["Temperature", "Humidity", "Rainfall"]


def main():
    log = json.loads((RESULTS / "panel" / "tuning.json").read_text())
    params = {(r["Target"], r["Model"]): r["best"] for r in log if "best" in r}
    p24 = pd.read_csv(DATA / "panel_monthly_24st.csv")
    common24 = set(p24.Station.unique())
    p24[p24.Year >= 1999].to_csv(DATA / "_panel24_1999.csv", index=False)
    configs = {"A: 24 st., 1981-2022": DATA / "panel_monthly_24st.csv",
               "B: 24 st., 1999-2022": DATA / "_panel24_1999.csv",
               "C: 33 st., 1999-2022": DATA / "panel_monthly_33st.csv"}
    rows = []
    for name, path in configs.items():
        raw, df = pf.load_panel(path, DYN3, TARGETS)
        for target in TARGETS:
            feats = tabular_features(target, DYN3)
            d = add_fold_features(df, target, df.Year < 2018)
            tr, te = d[d.Year < 2018], d[d.Year >= 2018]
            evals = [(name, te)]
            if name.startswith("C"):
                evals.append(("C24: 33-st. model on 24 common st.", te[te.Station.isin(common24)]))
            for lab, t in evals:
                rows.append({"Config": lab, "Target": target, "Model": "Climatology", "Seed": 0,
                             "n_train": len(tr), "n_stations_train": tr.Station.nunique(),
                             "train_years": f"{tr.Year.min()}-{tr.Year.max()}",
                             **metrics(t[target], t.clim, t.clim)})
            for mdl in ["Ridge", "RF", "XGB"]:
                for sd in ([0] if mdl == "Ridge" else [0, 1, 2]):
                    est = make_estimator(mdl, params[(target, mdl)], sd).fit(tr[feats].to_numpy(float), tr[target].to_numpy(float))
                    for lab, t in evals:
                        p = est.predict(t[feats].to_numpy(float))
                        if target == "Rainfall":
                            p = np.clip(p, 0, None)
                        rows.append({"Config": lab, "Target": target, "Model": mdl, "Seed": sd, "n_train": len(tr),
                                     "n_stations_train": tr.Station.nunique(),
                                     "train_years": f"{tr.Year.min()}-{tr.Year.max()}",
                                     **metrics(t[target], p, t.clim)})
            print(name, target, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "interval_comparison.csv", index=False)


if __name__ == "__main__":
    main()
