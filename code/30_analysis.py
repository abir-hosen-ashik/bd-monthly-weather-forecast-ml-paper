import glob

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf

from common import DATA, RESULTS, SEASON_OF_MONTH, TARGETS, TEST_YEARS, metrics

PANEL = RESULTS / "panel"
TAB = RESULTS / "tables"
TAB.mkdir(exist_ok=True, parents=True)
ORDER = ["Climatology", "Persistence", "Seasonal naive", "SARIMA", "Ridge", "DT", "RF", "SVR", "KNN",
         "XGB", "MLP", "LSTM", "GRU"]


def load(panel_dir=PANEL):
    files = [f for f in glob.glob(str(panel_dir / "*__*.csv.gz")) if "/inner_" not in f]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["Season"] = df["Month"].map(SEASON_OF_MONTH)
    df["t"] = df["Year"] * 12 + df["Month"] - 1
    return df


def per_seed_metrics(g):
    return pd.DataFrame([{"Seed": s, **metrics(h.y, h.pred, h.clim)} for s, h in g.groupby("Seed")])


def summarise(df, by):
    rows = []
    for key, g in df.groupby(by):
        m = per_seed_metrics(g)
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec["n_seeds"] = len(m)
        for c in ["MAE", "RMSE", "Bias", "R2", "MSESS_clim", "r"]:
            rec[c] = m[c].mean()
            rec[c + "_sd"] = m[c].std(ddof=1) if len(m) > 1 else 0.0
        if len(m) > 1:
            h = stats.t.ppf(0.975, len(m) - 1) * m["RMSE"].std(ddof=1) / np.sqrt(len(m))
            rec["RMSE_CI95"] = f"[{m.RMSE.mean() - h:.3f}, {m.RMSE.mean() + h:.3f}]"
        rec["N"] = int(m["N"].iloc[0])
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["Model"] = pd.Categorical(out["Model"], ORDER, ordered=True)
    return out.sort_values(by).reset_index(drop=True)


def seed_mean(df):
    return df.groupby(["Target", "Model", "Station", "Division", "Year", "Month", "t", "Season"], as_index=False, observed=True).agg(
        y=("y", "first"), clim=("clim", "first"), clim_sd=("clim_sd", "first"), pred=("pred", "mean"))


def best_ml(main):
    ml = main[~main.Model.isin(["Climatology", "Persistence", "Seasonal naive", "SARIMA"])]
    return {t: str(g.sort_values("RMSE").iloc[0].Model) for t, g in ml.groupby("Target")}


def dm_test(d, h=1):
    d = np.asarray(d, float)
    T = len(d)
    dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / T
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * stats.t.sf(abs(hln), T - 1)
    return hln, p


def dm_tables(df, best):
    rows = []
    for target, g in df.groupby("Target"):
        g = g.assign(se=(g.pred - g.y) ** 2)
        loss = g.groupby(["Model", "t"], observed=True)["se"].mean().unstack(0)
        ref = best[target]
        for other in [m for m in ORDER if m in loss.columns and m != ref]:
            d = (loss[other] - loss[ref]).dropna()
            stat, p = dm_test(d.to_numpy())
            rows.append({"Target": target, "Reference": ref, "Competitor": other, "T": len(d),
                         "mean_loss_diff": d.mean(), "DM_HLN": stat, "p_value": p})
        for other in [m for m in ORDER if m in loss.columns and m != "Climatology"]:
            d = (loss["Climatology"] - loss[other]).dropna()
            stat, p = dm_test(d.to_numpy())
            rows.append({"Target": target, "Reference": other, "Competitor": "Climatology", "T": len(d),
                         "mean_loss_diff": d.mean(), "DM_HLN": stat, "p_value": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for (t, r), idx in out.groupby(["Target", out.Competitor == "Climatology"]).groups.items():
        out.loc[idx, "p_holm"] = multipletests(out.loc[idx, "p_value"], method="holm")[1]
    return out


def extremes(sm):
    raw = pd.read_csv(DATA / "panel_monthly_24st.csv")
    thr = raw[raw.Year < TEST_YEARS[0]].groupby("Station")["Rainfall"].quantile([0.90, 0.95]).unstack()
    rows = []
    r = sm[sm.Target == "Rainfall"].copy()
    for q in [0.90, 0.95]:
        r["thr"] = r.Station.map(thr[q])
        for mdl, g in r.groupby("Model", observed=True):
            ext = g[g.y >= g.thr]
            obs, fc = g.y >= g.thr, g.pred >= g.thr
            hits, miss, fa = int((obs & fc).sum()), int((obs & ~fc).sum()), int((~obs & fc).sum())
            rows.append({"quantile": q, "Model": mdl, "n_extreme": len(ext),
                         "MAE": np.mean(np.abs(ext.pred - ext.y)), "RMSE": np.sqrt(np.mean((ext.pred - ext.y) ** 2)),
                         "Bias": np.mean(ext.pred - ext.y), "RelBias_%": 100 * np.mean(ext.pred - ext.y) / np.mean(ext.y),
                         "POD": hits / max(hits + miss, 1), "FAR": fa / max(hits + fa, 1),
                         "CSI": hits / max(hits + miss + fa, 1)})
    out = pd.DataFrame(rows)
    out["Model"] = pd.Categorical(out["Model"], ORDER, ordered=True)
    return out.sort_values(["quantile", "Model"])


def residual_diag(sm, best):
    rows, acfs = [], []
    for target, mdl in best.items():
        for m in [mdl, "Climatology"]:
            g = sm[(sm.Target == target) & (sm.Model == m)].sort_values(["Station", "t"])
            e = (g.pred - g.y).to_numpy()
            slope = stats.linregress(g.y, e).slope
            lb_p, ac = [], []
            for st, h in g.groupby("Station"):
                res = (h.pred - h.y).to_numpy()
                if len(res) < 24:
                    continue
                lb_p.append(acorr_ljungbox(res, lags=[12], return_df=True)["lb_pvalue"].iloc[0])
                ac.append(acf(res, nlags=12, fft=False)[1:])
            ac = np.mean(ac, axis=0)
            rows.append({"Target": target, "Model": m, "mean_residual": e.mean(), "sd_residual": e.std(),
                         "skewness": stats.skew(e), "cond_bias_slope_resid_vs_obs": slope,
                         "acf_lag1_mean": ac[0], "acf_lag12_mean": ac[11],
                         "stations_LjungBox12_p<0.05": int(np.sum(np.array(lb_p) < 0.05)), "n_stations": len(lb_p)})
            acfs += [{"Target": target, "Model": m, "lag": k + 1, "acf": v} for k, v in enumerate(ac)]
    return pd.DataFrame(rows), pd.DataFrame(acfs)


def conformal(sm, models_for_pi):
    rows, series = [], []
    for target in TARGETS:
        for mdl in models_for_pi[target]:
            inner = pd.read_csv(PANEL / f"inner_{target}__{mdl}.csv.gz")
            inner["score"] = np.abs(inner[target] - inner.pred) / inner.clim_sd
            inner["Season"] = inner.Month.map(SEASON_OF_MONTH)
            g = sm[(sm.Target == target) & (sm.Model == mdl)].copy()
            g["score"] = np.abs(g.y - g.pred) / g.clim_sd
            for alpha in [0.1, 0.2]:
                out = []
                for Y in TEST_YEARS:
                    cal = pd.concat([inner[["Season", "score"]], g[g.Year < Y][["Season", "score"]]])
                    te = g[g.Year == Y].copy()
                    q = {}
                    for s, c in cal.groupby("Season"):
                        n = len(c)
                        q[s] = np.quantile(c.score, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
                    te["lo"] = te.pred - te.Season.map(q) * te.clim_sd
                    te["hi"] = te.pred + te.Season.map(q) * te.clim_sd
                    if target == "Rainfall":
                        te["lo"] = te["lo"].clip(lower=0)
                    out.append(te)
                o = pd.concat(out)
                cover = (o.y >= o.lo) & (o.y <= o.hi)
                width = o.hi - o.lo
                wink = width + (2 / alpha) * ((o.lo - o.y).clip(lower=0) + (o.y - o.hi).clip(lower=0))
                rows.append({"Target": target, "Model": mdl, "nominal": 1 - alpha, "Season": "All",
                             "PICP": cover.mean(), "mean_width": width.mean(), "interval_score": wink.mean()})
                for s, h in o.assign(cover=cover, width=width).groupby("Season"):
                    rows.append({"Target": target, "Model": mdl, "nominal": 1 - alpha, "Season": s,
                                 "PICP": h.cover.mean(), "mean_width": h.width.mean()})
                if alpha == 0.1:
                    series.append(o.assign(Model=mdl, Target=target)[["Target", "Model", "Station", "Year", "Month", "y", "pred", "lo", "hi"]])
    pd.concat(series).to_csv(TAB / "intervals_series.csv.gz", index=False)
    return pd.DataFrame(rows)


def main():
    df = load()
    main_tab = summarise(df, ["Target", "Model"])
    main_tab.to_csv(TAB / "main_metrics.csv", index=False)
    best = best_ml(main_tab)
    print("best ML model per target:", best)
    summarise(df, ["Target", "Model", "Year"]).to_csv(TAB / "per_year.csv", index=False)
    summarise(df, ["Target", "Model", "Season"]).to_csv(TAB / "per_season.csv", index=False)
    summarise(df, ["Target", "Model", "Division"]).to_csv(TAB / "per_division.csv", index=False)
    sm = seed_mean(df)
    extremes(sm).to_csv(TAB / "extremes.csv", index=False)
    dm_tables(df, best).to_csv(TAB / "dm_tests.csv", index=False)
    rd, ra = residual_diag(sm, best)
    rd.to_csv(TAB / "residual_diagnostics.csv", index=False)
    ra.to_csv(TAB / "residual_acf.csv", index=False)
    pis = {t: sorted({best[t], "RF", "XGB"}) for t in TARGETS}
    conformal(sm, pis).to_csv(TAB / "intervals.csv", index=False)
    tp = PANEL / "timing.csv"
    if tp.exists():
        tm = pd.read_csv(tp)
        tm.groupby(["Model"])["fit_predict_seconds"].agg(["mean", "std", "count"]).to_csv(TAB / "timing.csv")
    pd.Series(best).to_json(TAB / "best_models.json")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(main_tab[["Target", "Model", "MAE", "RMSE", "RMSE_sd", "Bias", "R2", "MSESS_clim"]].round(3).to_string())


if __name__ == "__main__":
    main()
