import argparse
import json

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import (DATA, DL_SEEDS, INNER_VAL_BLOCKS, RESULTS, SEEDS, TARGETS, TEST_YEARS, Timer)
from features import (add_fold_features, add_lags, add_neighbour_lag, sequence_arrays,
                      static_features, tabular_features, DYN)
from models import BASELINES, ML, SEQUENCE, STOCHASTIC, TORCH, fit_predict, sample_configs

OUT = RESULTS / "panel"
OUT.mkdir(exist_ok=True, parents=True)
KEEP = ["Station", "Division", "Year", "Month"]


def load_panel(path=DATA / "panel_monthly_24st.csv", dyn=DYN, targets=TARGETS):
    raw = pd.read_csv(path)
    df = add_lags(raw, targets, dyn)
    for tg in targets:
        df = add_neighbour_lag(df, tg, df["Station"].unique())
    return raw, df


def design(df, target, name, dyn):
    if name in SEQUENCE:
        return sequence_arrays(df, dyn), df[static_features(target)].to_numpy(float)
    return df[tabular_features(target, dyn)].to_numpy(float), None


def val_mask_for(dtr, years=3):
    last = dtr["Year"].max()
    return (dtr["Year"] > last - years).to_numpy()


def run_model(name, params, seed, dtr, dte, target, dyn):
    Xtr, Str = design(dtr, target, name, dyn)
    Xte, Ste = design(dte, target, name, dyn)
    vm = val_mask_for(dtr) if name in TORCH else None
    with Timer() as t:
        pred, est = fit_predict(name, params, seed, Xtr, dtr[target].to_numpy(float), Xte, Str, Ste, vm)
    if target == "Rainfall":
        pred = np.clip(pred, 0, None)
    return pred, t.dt, est


def tune(name, df, target, dyn, log):
    dev = df[df["Year"] <= 2017]
    best, best_score, best_inner = None, np.inf, None
    blocks = INNER_VAL_BLOCKS[-2:] if name in SEQUENCE else INNER_VAL_BLOCKS
    for cfg in sample_configs(name):
        preds, sq = [], []
        for a, b in blocks:
            trm = dev["Year"] < a
            d = add_fold_features(dev, target, trm)
            dtr, dva = d[trm], d[(d["Year"] >= a) & (d["Year"] <= b)]
            p, _, _ = run_model(name, cfg, 0, dtr, dva, target, dyn)
            sq.append((p - dva[target].to_numpy()) ** 2)
            preds.append(dva[KEEP + [target, "clim", "clim_sd"]].assign(pred=p))
        score = float(np.mean(np.concatenate(sq)))
        log.append({"Target": target, "Model": name, "params": cfg, "inner_MSE": score})
        print(f"    {name:5s} {cfg} inner RMSE={np.sqrt(score):.3f}", flush=True)
        if score < best_score:
            best, best_score, best_inner = cfg, score, pd.concat(preds)
    return best, best_score, best_inner


def sarima_forecasts(raw, target, years):
    rows = []
    for st, s in raw.groupby("Station"):
        s = s.copy()
        s["date"] = pd.PeriodIndex.from_fields(year=s["Year"], month=s["Month"], freq="M")
        y = s.set_index("date")[target].sort_index()
        y = y.reindex(pd.period_range(y.index.min(), y.index.max(), freq="M"))
        yt = np.log1p(y) if target == "Rainfall" else y
        train = yt[yt.index.year < years[0]]
        try:
            res = SARIMAX(train, order=(1, 0, 0), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
            full = res.apply(yt)
            pr = full.get_prediction().predicted_mean
        except Exception as e:
            print("SARIMA failed", st, e)
            continue
        pr = pr[pr.index.year >= years[0]]
        if target == "Rainfall":
            pr = np.clip(np.expm1(pr), 0, None)
        rows.append(pd.DataFrame({"Station": st, "Year": pr.index.year, "Month": pr.index.month, "pred": pr.values}))
    return pd.concat(rows)


def main(targets, models, dyn=DYN, tag="", panel_path=DATA / "panel_monthly_24st.csv"):
    raw, df = load_panel(panel_path, dyn, targets)
    out = OUT if not tag else RESULTS / f"panel_{tag}"
    out.mkdir(exist_ok=True, parents=True)
    tlog_path = out / "tuning.json"
    tlog = json.loads(tlog_path.read_text()) if tlog_path.exists() else []
    timing = []
    print(f"panel rows after lag construction: {len(df)}; stations {df.Station.nunique()}", flush=True)
    for target in targets:
        for name in models:
            path = out / f"{target}__{name}.csv.gz"
            if path.exists():
                print(f"skip {path.name}")
                continue
            print(f"[{target}] {name}", flush=True)
            parts = []
            params, inner = {}, None
            if name in ML:
                with Timer() as tt:
                    params, score, inner = tune(name, df, target, dyn, tlog)
                tlog.append({"Target": target, "Model": name, "best": params, "best_inner_MSE": score,
                             "tuning_seconds": tt.dt})
                tlog_path.write_text(json.dumps(tlog, indent=1, default=str))
                inner.assign(Model=name, Target=target).to_csv(out / f"inner_{target}__{name}.csv.gz", index=False)
            if name == "SARIMA":
                sp = sarima_forecasts(raw, target, TEST_YEARS)
            for Y in TEST_YEARS:
                trm = df["Year"] < Y
                d = add_fold_features(df, target, trm)
                dtr, dte = d[trm], d[d["Year"] == Y]
                base = dte[KEEP + [target, "clim", "clim_sd"]].rename(columns={target: "y"})
                if name == "Climatology":
                    parts.append(base.assign(pred=dte["clim"].to_numpy(), Seed=0))
                elif name == "Persistence":
                    parts.append(base.assign(pred=dte[f"{target}_raw_lag1"].to_numpy(), Seed=0))
                elif name == "Seasonal naive":
                    parts.append(base.assign(pred=dte[f"{target}_raw_lag12"].to_numpy(), Seed=0))
                elif name == "SARIMA":
                    m = base.merge(sp, on=["Station", "Year", "Month"], how="left")
                    m["pred"] = m["pred"].fillna(m["clim"])
                    parts.append(m.assign(Seed=0))
                else:
                    seeds = (DL_SEEDS if name in SEQUENCE else SEEDS) if name in STOCHASTIC else [0]
                    for sd in seeds:
                        p, dt, _ = run_model(name, params, sd, dtr, dte, target, dyn)
                        timing.append({"Target": target, "Model": name, "Year": Y, "Seed": sd,
                                       "fit_predict_seconds": dt, "n_train": len(dtr)})
                        parts.append(base.assign(pred=p, Seed=sd))
            res = pd.concat(parts).assign(Model=name, Target=target)
            res.to_csv(path, index=False)
            rm = res.groupby("Seed").apply(lambda g: np.sqrt(np.mean((g.pred - g.y) ** 2)))
            print(f"   -> test RMSE {rm.mean():.3f} ± {rm.std():.3f}", flush=True)
            if timing:
                tp = out / "timing.csv"
                pd.DataFrame(timing).to_csv(tp, mode="a", header=not tp.exists(), index=False)
                timing = []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--models", nargs="+", default=BASELINES + ML)
    a = ap.parse_args()
    main(a.targets, a.models)
