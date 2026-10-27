import numpy as np
import pandas as pd
from scipy import stats

from common import DATA, RESULTS, SEASON_OF_MONTH

OUT = RESULTS / "trends"
OUT.mkdir(exist_ok=True, parents=True)


def mann_kendall(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18
    slopes = [(x[j] - x[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    sen = np.median(slopes)
    det = x - sen * np.arange(n)
    r1 = np.corrcoef(det[:-1], det[1:])[0, 1]
    if r1 > 0:
        nstar = 1 + 2 * sum((1 - k / n) * r1 ** k for k in range(1, n))
        var *= nstar
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return s, z, p, sen


def complete_years(p, months_needed):
    n = p.groupby(["Station", "Year"])["Month"].transform("count")
    return p[n == months_needed]


def main():
    p = pd.read_csv(DATA / "panel_monthly_24st.csv")
    p["Season"] = p.Month.map(SEASON_OF_MONTH)
    rows = []
    agg = {"Temperature": "mean", "Humidity": "mean", "Rainfall": "sum"}
    st_year = complete_years(p, 12).groupby(["Station", "Year"]).agg(agg).reset_index()
    nat = st_year.groupby("Year")[list(agg)].mean()
    for v in agg:
        s, z, pv, sen = mann_kendall(nat[v].to_numpy())
        rows.append({"Scope": "Bangladesh", "Period": "Annual", "Variable": v, "MK_S": s, "Z": z, "p": pv,
                     "Sen_slope_per_decade": 10 * sen, "years": f"{nat.index.min()}-{nat.index.max()}"})
    for season, g in p.groupby("Season"):
        g = complete_years(g, g.groupby(["Station", "Year"])["Month"].count().max())
        sy = g.groupby(["Station", "Year"]).agg(agg).reset_index().groupby("Year")[list(agg)].mean()
        for v in agg:
            s, z, pv, sen = mann_kendall(sy[v].to_numpy())
            rows.append({"Scope": "Bangladesh", "Period": season, "Variable": v, "MK_S": s, "Z": z, "p": pv,
                         "Sen_slope_per_decade": 10 * sen, "years": f"{sy.index.min()}-{sy.index.max()}"})
    for div, g in complete_years(p, 12).groupby("Division"):
        dy = g.groupby(["Station", "Year"]).agg(agg).reset_index().groupby("Year")[list(agg)].mean()
        for v in agg:
            s, z, pv, sen = mann_kendall(dy[v].to_numpy())
            rows.append({"Scope": div, "Period": "Annual", "Variable": v, "MK_S": s, "Z": z, "p": pv,
                         "Sen_slope_per_decade": 10 * sen, "years": f"{dy.index.min()}-{dy.index.max()}"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mann_kendall.csv", index=False)
    nat.to_csv(OUT / "national_annual_series.csv")
    print(out[out.Scope == "Bangladesh"].round(3).to_string())


if __name__ == "__main__":
    main()
