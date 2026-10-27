import numpy as np
import openpyxl
import pandas as pd

from common import (DATA, SRC_DAILY, SRC_OTHER_DAILY, SRC_PANEL_1980, SRC_PANEL_1999,
                    STATION_ALIASES, STATION_COORDS, STATION_DIVISION, haversine_km, save_json)

RANGES_MONTHLY = {"Temperature": (12, 36), "Humidity": (30, 100), "Rainfall": (0, 3000),
                  "Pressure": (97, 104), "Wind_speed": (0, 60), "Dew_Point": (-5, 35)}
RANGES_DAILY = {"Pressure": (94, 106), "Wind_speed": (0, 150), "Wind_Direction": (0, 360),
                "Dew_Point": (-10, 35)}
ROBUST_VARS = ["Temperature", "Humidity", "Pressure", "Wind_speed", "Dew_Point"]
ROBUST_Z = 5.0
AUX = ["Pressure", "Wind_speed", "Wind_dir_sin", "Wind_dir_cos", "Dew_Point"]
qc = {}


def harmonise(df):
    df["Station"] = df["Station"].astype(str).str.strip().replace(STATION_ALIASES)
    return df


def range_check(df, ranges, tag):
    for col, (lo, hi) in ranges.items():
        if col not in df:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = (df[col] < lo) | (df[col] > hi)
        qc[f"{tag}_range_flagged_{col}"] = int(bad.sum())
        df.loc[bad, col] = np.nan
    return df


def robust_screen(df, cols, tag):
    for col in cols:
        if col not in df:
            continue
        g = df.groupby(["Station", "Month"])[col]
        med = g.transform("median")
        mad = g.transform(lambda s: np.median(np.abs(s - np.median(s.dropna()))) if s.notna().any() else np.nan)
        z = 0.6745 * (df[col] - med) / mad.replace(0, np.nan)
        bad = z.abs() > ROBUST_Z
        qc[f"{tag}_robust_flagged_{col}"] = int(bad.sum())
        df.loc[bad, col] = np.nan
    return df


def load_other_daily():
    cache = DATA / "_cache_other_daily.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    wb = openpyxl.load_workbook(SRC_OTHER_DAILY, read_only=True, data_only=True)
    rows = [r for r in wb["other_data"].iter_rows(min_row=2, values_only=True) if r[0] is not None]
    d = pd.DataFrame(rows, columns=["Station", "Year", "Month", "Day", "Pressure", "Wind_speed",
                                    "Wind_Direction", "Dew_Point"])
    d.to_csv(cache, index=False)
    return d


def other_monthly():
    d = harmonise(load_other_daily())
    for c in ["Pressure", "Wind_speed", "Wind_Direction", "Dew_Point"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    qc["daily_aux_records"] = len(d)
    qc["daily_aux_missing_before_qc"] = int(d[["Pressure", "Wind_speed", "Wind_Direction", "Dew_Point"]].isna().sum().sum())
    d = range_check(d, RANGES_DAILY, "daily_aux")
    d["Wind_dir_sin"] = np.sin(np.deg2rad(d["Wind_Direction"]))
    d["Wind_dir_cos"] = np.cos(np.deg2rad(d["Wind_Direction"]))
    agg = {c: (c, "mean") for c in ["Pressure", "Wind_speed", "Wind_dir_sin", "Wind_dir_cos", "Dew_Point"]}
    agg.update({f"n_{c}": (c, "count") for c in ["Pressure", "Wind_speed", "Wind_dir_sin", "Dew_Point"]})
    m = d.groupby(["Station", "Year", "Month"], as_index=False).agg(**agg)
    insufficient = 0
    for c, nc in [("Pressure", "n_Pressure"), ("Wind_speed", "n_Wind_speed"),
                  ("Wind_dir_sin", "n_Wind_dir_sin"), ("Wind_dir_cos", "n_Wind_dir_sin"),
                  ("Dew_Point", "n_Dew_Point")]:
        bad = m[nc] < 20
        insufficient += int(bad.sum())
        m.loc[bad, c] = np.nan
    qc["monthly_aux_values_with_<20_valid_days"] = insufficient
    return m.drop(columns=[c for c in m if c.startswith("n_")])


def temporal_interpolate(df, cols, tag, limit=2):
    df = df.sort_values(["Station", "Year", "Month"]).copy()
    n_before = int(df[cols].isna().sum().sum())
    df[cols] = df.groupby("Station")[cols].transform(
        lambda s: s.interpolate(method="linear", limit=limit, limit_area="inside"))
    qc[f"{tag}_temporal_interpolated"] = n_before - int(df[cols].isna().sum().sum())
    return df


def spatial_idw(df, cols, tag, radius_km=100, min_nb=2):
    filled = 0
    stations = df["Station"].unique()
    dist = {(a, b): haversine_km(*STATION_COORDS[a], *STATION_COORDS[b]) for a in stations for b in stations}
    piv = {c: df.pivot_table(index=["Year", "Month"], columns="Station", values=c) for c in cols}
    for c in cols:
        P = piv[c]
        for idx in df.index[df[c].isna()]:
            s, y, mo = df.at[idx, "Station"], df.at[idx, "Year"], df.at[idx, "Month"]
            if (y, mo) not in P.index:
                continue
            row = P.loc[(y, mo)]
            nb = [(b, row[b]) for b in stations if b != s and dist[(s, b)] <= radius_km and pd.notna(row.get(b))]
            if len(nb) >= min_nb:
                w = np.array([1 / dist[(s, b)] ** 2 for b, _ in nb])
                v = np.array([val for _, val in nb])
                df.at[idx, c] = float(np.sum(w * v) / np.sum(w))
                filled += 1
    qc[f"{tag}_spatial_idw_filled"] = filled
    return df


def build_24():
    p = harmonise(pd.read_csv(SRC_PANEL_1980))
    qc["panel1980_raw_rows"] = len(p)
    qc["panel1980_raw_stations"] = p["Station"].nunique()
    dup = p.duplicated(["Station", "Year", "Month"])
    qc["panel1980_duplicates_removed"] = int(dup.sum())
    p = p[~dup]
    qc["panel1980_missing_targets_raw"] = int(p[["Temperature", "Humidity", "Rainfall"]].isna().sum().sum())
    p = range_check(p, RANGES_MONTHLY, "panel1980")
    p = robust_screen(p, ["Temperature", "Humidity"], "panel1980")

    m = other_monthly()
    df = p.merge(m, on=["Station", "Year", "Month"], how="left")
    df = df[df["Year"] >= 1981]
    has_aux = df.groupby("Station")["Pressure"].apply(lambda s: s.notna().mean() > 0.5)
    qc["stations_dropped_no_aux_record"] = sorted(has_aux.index[~has_aux].tolist())
    df = df[df["Station"].isin(has_aux.index[has_aux])]
    df = range_check(df, {k: RANGES_MONTHLY[k] for k in ["Pressure", "Wind_speed", "Dew_Point"]}, "panel24_aux")
    df = robust_screen(df, ["Pressure", "Wind_speed", "Dew_Point"], "panel24_aux")
    bad_td = df["Dew_Point"] > df["Temperature"] + 1.0
    qc["panel24_dewpoint_gt_T_flagged"] = int(bad_td.sum())
    df.loc[bad_td, "Dew_Point"] = np.nan
    qc["panel24_missing_before_imputation"] = {c: int(df[c].isna().sum()) for c in ["Temperature", "Humidity", "Rainfall"] + AUX}

    df = temporal_interpolate(df, ["Temperature", "Humidity", "Rainfall"] + AUX, "panel24")
    df = spatial_idw(df, AUX, "panel24_aux")
    n0 = len(df)
    df = df.dropna(subset=["Temperature", "Humidity", "Rainfall"] + AUX)
    qc["panel24_rows_dropped_incomplete"] = n0 - len(df)
    df = df.sort_values(["Station", "Year", "Month"]).reset_index(drop=True)
    df["Division"] = df["Station"].map(STATION_DIVISION)
    df["Latitude"] = df["Station"].map(lambda s: STATION_COORDS[s][0])
    df["Longitude"] = df["Station"].map(lambda s: STATION_COORDS[s][1])
    qc["panel24_final_rows"] = len(df)
    qc["panel24_final_stations"] = df["Station"].nunique()
    qc["panel24_station_list"] = sorted(df["Station"].unique().tolist())
    qc["panel24_period"] = f"{df.Year.min()}-{df.Year.max()}"
    df.to_csv(DATA / "panel_monthly_24st.csv", index=False)
    return df


def build_33():
    p = harmonise(pd.read_csv(SRC_PANEL_1999))
    p = p.rename(columns={"Monthly_Total_Rainfall": "Rainfall", "Average_Temperature": "Temperature"})
    p = p[["Station", "Year", "Month", "Temperature", "Humidity", "Rainfall"]]
    for c in ["Temperature", "Humidity", "Rainfall"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    qc["panel33_raw_rows"] = len(p)
    p = p[~p.duplicated(["Station", "Year", "Month"])]
    p = range_check(p, RANGES_MONTHLY, "panel33")
    p = robust_screen(p, ["Temperature", "Humidity"], "panel33")
    p = temporal_interpolate(p, ["Temperature", "Humidity", "Rainfall"], "panel33")
    n0 = len(p)
    p = p.dropna()
    qc["panel33_rows_dropped_incomplete"] = n0 - len(p)
    p["Division"] = p["Station"].map(STATION_DIVISION)
    p["Latitude"] = p["Station"].map(lambda s: STATION_COORDS[s][0])
    p["Longitude"] = p["Station"].map(lambda s: STATION_COORDS[s][1])
    qc["panel33_final_rows"] = len(p)
    qc["panel33_final_stations"] = p["Station"].nunique()
    p.to_csv(DATA / "panel_monthly_33st.csv", index=False)


def build_daily():
    d = harmonise(pd.read_csv(SRC_DAILY))
    d = d.rename(columns={"Average_Temperature": "Temperature"})
    qc["daily_raw_rows"] = len(d)
    rng = {"Temperature": (0, 45), "Max_Temperature": (0, 48), "Min_Temperature": (-2, 40),
           "Humidity": (5, 100), "Pressure": (94, 106), "Wind_speed": (0, 150), "Dew_Point": (-10, 35)}
    d = range_check(d, rng, "daily")
    bad = d["Min_Temperature"] > d["Max_Temperature"]
    qc["daily_tmin_gt_tmax_flagged"] = int(bad.sum())
    d.loc[bad, ["Min_Temperature", "Max_Temperature", "Temperature"]] = np.nan
    d = d.sort_values(["Station", "Year", "Month", "Day"])
    cols = ["Temperature", "Humidity", "Pressure", "Wind_speed", "Dew_Point", "Wind_dir_sin", "Wind_dir_cos"]
    n_before = int(d[cols].isna().sum().sum())
    d[cols] = d.groupby("Station")[cols].transform(lambda s: s.interpolate(limit=3, limit_area="inside"))
    qc["daily_temporal_interpolated"] = n_before - int(d[cols].isna().sum().sum())
    n0 = len(d)
    d = d.dropna(subset=cols)
    qc["daily_rows_dropped_incomplete"] = n0 - len(d)
    qc["daily_final_rows"] = len(d)
    qc["daily_final_stations"] = d["Station"].nunique()
    d["Date"] = pd.to_datetime(dict(year=d.Year, month=d.Month, day=d.Day), errors="coerce")
    d = d.dropna(subset=["Date"])
    d[["Station", "Date", "Year", "Month", "Day"] + cols].to_csv(DATA / "daily_panel.csv", index=False)


if __name__ == "__main__":
    build_24()
    build_33()
    build_daily()
    save_json(qc, DATA / "qc_report.json")
    for k, v in qc.items():
        print(f"{k}: {v}")
