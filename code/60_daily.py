import numpy as np
import pandas as pd

from common import DATA, RESULTS, STATION_COORDS, metrics
from models import make_estimator

OUT = RESULTS / "daily"
OUT.mkdir(exist_ok=True, parents=True)
VARS = ["Temperature", "Humidity", "Pressure", "Wind_speed", "Dew_Point", "Wind_dir_sin", "Wind_dir_cos"]


def build(d):
    d = d.sort_values(["Station", "Date"]).reset_index(drop=True)
    d["Date"] = pd.to_datetime(d["Date"])
    g = d.groupby("Station")
    dd = g["Date"].diff().dt.days
    for k in (1, 2, 3, 7):
        ok = (d["Date"] - g["Date"].shift(k)).dt.days == k
        for v in VARS if k == 1 else ["Temperature", "Humidity"]:
            d[f"{v}_lag{k}"] = g[v].shift(k).where(ok)
    for v in ["Temperature", "Humidity"]:
        d[f"{v}_mean7"] = g[v].transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
    d["doy"] = d["Date"].dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * d.doy / 365.25), np.cos(2 * np.pi * d.doy / 365.25)
    d["Latitude"] = d.Station.map(lambda s: STATION_COORDS[s][0])
    d["Longitude"] = d.Station.map(lambda s: STATION_COORDS[s][1])
    return d.dropna().reset_index(drop=True)


def clim_doy(d, target, train_mask):
    tr = d[train_mask]
    c = tr.groupby(["Station", "doy"])[target].mean().unstack()
    c = c.T.reindex(range(1, 367)).interpolate(limit_direction="both")
    c = pd.concat([c.iloc[-15:], c, c.iloc[:15]]).rolling(31, center=True).mean().iloc[15:-15]
    return c.stack().rename("clim").reset_index().rename(columns={"level_0": "doy"})


DAILY_PARAMS = {
    "Ridge": {"alpha": 1.0},
    "RF": {"max_depth": 20, "min_samples_leaf": 5, "max_features": 0.5},
    "XGB": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_lambda": 1},
}


def main():
    d = build(pd.read_csv(DATA / "daily_panel.csv"))
    rows, preds = [], []
    for target in ["Temperature", "Humidity"]:
        trm = d.Year < 2018
        cl = clim_doy(d, target, trm)
        x = d.merge(cl, on=["Station", "doy"], how="left")
        other = [f"{v}_lag1" for v in VARS if v != target]
        feats = [f"{target}_lag{k}" for k in (1, 2, 3, 7)] + [f"{target}_mean7"] + other + \
                ["doy_sin", "doy_cos", "Latitude", "Longitude", "clim"]
        tr, te = x[x.Year < 2018], x[x.Year >= 2018]
        y = te[target].to_numpy()
        rows.append({"Target": target, "Model": "Climatology", "Seed": 0, **metrics(y, te.clim, te.clim)})
        rows.append({"Target": target, "Model": "Persistence", "Seed": 0, **metrics(y, te[f"{target}_lag1"], te.clim)})
        for mdl in ["Ridge", "RF", "XGB"]:
            p = dict(DAILY_PARAMS[mdl])
            for sd in ([0] if mdl == "Ridge" else [0, 1, 2]):
                est = make_estimator(mdl, p, sd)
                if mdl == "RF":
                    est.set_params(model__n_estimators=100, model__max_samples=0.25)
                est.fit(tr[feats].to_numpy(float), tr[target].to_numpy(float))
                pr = est.predict(te[feats].to_numpy(float))
                rows.append({"Target": target, "Model": mdl, "Seed": sd, **metrics(y, pr, te.clim)})
                if sd == 0:
                    preds.append(te[["Station", "Date", target]].rename(columns={target: "y"}).assign(Target=target, Model=mdl, pred=pr))
                print(target, mdl, sd, rows[-1]["RMSE"], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "daily_metrics.csv", index=False)
    pd.concat(preds).to_csv(OUT / "daily_predictions.csv.gz", index=False)


if __name__ == "__main__":
    main()
