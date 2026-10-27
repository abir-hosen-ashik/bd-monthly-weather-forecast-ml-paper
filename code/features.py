import numpy as np
import pandas as pd

from common import STATION_COORDS, haversine_km

DYN = ["Temperature", "Humidity", "Rainfall", "Pressure", "Wind_speed",
       "Wind_dir_sin", "Wind_dir_cos", "Dew_Point"]
SEQ_LEN = 12


def _rain_tf(x):
    return np.log1p(np.clip(x, 0, None))


def add_lags(df, targets, dyn=DYN):
    df = df.sort_values(["Station", "Year", "Month"]).reset_index(drop=True).copy()
    df["t"] = df["Year"] * 12 + df["Month"] - 1
    g = df.groupby("Station")
    dyn = [v for v in dyn if v in df]
    for v in dyn:
        base = _rain_tf(df[v]) if v == "Rainfall" else df[v]
        df["_b_" + v] = base
    for k in range(1, SEQ_LEN + 1):
        ok = (df["t"] - g["t"].shift(k)) == k
        for v in dyn:
            df[f"{v}_lag{k}"] = g["_b_" + v].shift(k).where(ok)
    df = df.drop(columns=[c for c in df if c.startswith("_b_")])
    for v in targets:
        for k in (1, 12):
            ok = (df["t"] - g["t"].shift(k)) == k
            df[f"{v}_raw_lag{k}"] = g[v].shift(k).where(ok)
    for v in targets:
        df[f"{v}_roll3"] = df[[f"{v}_lag{k}" for k in (1, 2, 3)]].mean(axis=1)
    if "Dew_Point" in dyn:
        df["DPD_lag1"] = df["Temperature_lag1"] - df["Dew_Point_lag1"]
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    need = [f"{v}_lag{k}" for v in dyn for k in range(1, SEQ_LEN + 1)]
    df = df.dropna(subset=need).reset_index(drop=True)
    return df


def neighbour_table(stations, k=4):
    out = {}
    for s in stations:
        d = sorted(((haversine_km(*STATION_COORDS[s], *STATION_COORDS[b]), b) for b in stations if b != s))[:k]
        w = np.array([1 / dd ** 2 for dd, _ in d])
        out[s] = ([b for _, b in d], w / w.sum())
    return out


def add_neighbour_lag(df, target, allowed_stations):
    nb = neighbour_table(sorted(set(allowed_stations) | set(df["Station"].unique())), k=4)
    allowed = set(allowed_stations)
    piv = df.pivot_table(index="t", columns="Station", values=f"{target}_lag1")
    vals = np.full(len(df), np.nan)
    for i, (s, t) in enumerate(zip(df["Station"].to_numpy(), df["t"].to_numpy())):
        names, w = nb[s]
        pairs = [(b, ww) for b, ww in zip(names, w) if b in allowed]
        if not pairs:
            continue
        row = piv.loc[t]
        v = np.array([row.get(b, np.nan) for b, _ in pairs])
        ww = np.array([x for _, x in pairs])
        m = ~np.isnan(v)
        if m.any():
            vals[i] = np.sum(v[m] * ww[m]) / np.sum(ww[m])
    df = df.copy()
    df[f"nb_{target}_lag1"] = vals
    df[f"nb_{target}_lag1"] = df[f"nb_{target}_lag1"].fillna(df[f"{target}_lag1"])
    return df


def add_fold_features(df, target, train_mask):
    tr = df.loc[train_mask]
    clim = tr.groupby(["Station", "Month"])[target].agg(["mean", "std"])
    mclim = tr.groupby("Month")[target].agg(["mean", "std"])
    key = pd.MultiIndex.from_arrays([df["Station"], df["Month"]])
    mean = clim["mean"].reindex(key).to_numpy()
    sd = clim["std"].reindex(key).to_numpy()
    mean = np.where(np.isnan(mean), mclim["mean"].reindex(df["Month"]).to_numpy(), mean)
    sd = np.where(np.isnan(sd), mclim["std"].reindex(df["Month"]).to_numpy(), sd)
    out = df.copy()
    out["clim"] = mean
    out["clim_sd"] = np.maximum(sd, 1e-6)
    return out


def tabular_features(target, dyn):
    f = [f"{target}_lag{k}" for k in (1, 2, 3, 12)] + [f"{target}_roll3"]
    f += [f"{v}_lag1" for v in dyn if v != target]
    if "Dew_Point" in dyn:
        f += ["DPD_lag1"]
    f += ["Month_sin", "Month_cos", "Latitude", "Longitude", f"nb_{target}_lag1", "clim"]
    return f


def sequence_arrays(df, dyn):
    seq = []
    for k in range(SEQ_LEN, 0, -1):
        cols = [f"{v}_lag{k}" for v in dyn]
        mon = ((df["Month"] - k - 1) % 12) + 1
        step = np.column_stack([df[cols].to_numpy(float),
                                np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12)])
        seq.append(step)
    return np.stack(seq, axis=1)


def static_features(target):
    return ["Month_sin", "Month_cos", "Latitude", "Longitude", f"nb_{target}_lag1", "clim"]
