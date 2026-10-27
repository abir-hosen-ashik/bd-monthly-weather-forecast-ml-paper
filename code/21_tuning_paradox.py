import importlib

import numpy as np
import pandas as pd

from common import RESULTS, SRC_NATIONAL_1980, r2

nat = importlib.import_module("20_national")


def main():
    df = pd.read_csv(SRC_NATIONAL_1980).sort_values(["Year", "Month"]).reset_index(drop=True)
    best = pd.read_csv(RESULTS / "national" / "national_best_params.csv")
    best = best[best.Series.str.startswith("1980")]
    dev = df[df.Year <= 2017].reset_index(drop=True)
    rows = []
    for _, b in best.iterrows():
        t, m = b.Target, b.Model
        tuned = eval(b.params)
        X, y = nat.corrected_features(df, t), df[t].to_numpy(float)
        tr, te = (df.Year < 2018).to_numpy(), (df.Year >= 2018).to_numpy()
        var_dev = []
        for a, bb in nat.INNER_VAL_BLOCKS:
            var_dev.append(dev[(dev.Year >= a) & (dev.Year <= bb)][t].to_numpy())
        v = np.var(np.concatenate(var_dev))
        for lab, p in [("manual (Table 4a)", nat.MANUAL[t][m]), ("tuned (rolling-origin)", tuned)]:
            inner = nat.inner_score(dev, t, m, p)
            seeds = range(10) if m in ("RF", "MLP", "XGB") else [0]
            te_r2 = [r2(y[te], nat.corrected_est(m, p, s).fit(X[tr], y[tr]).predict(X[te])) for s in seeds]
            rows.append({"Target": t, "Model": m, "Config": lab, "params": str(p), "validation_MSE": inner,
                         "validation_R2": 1 - inner / v, "test_R2_mean": np.mean(te_r2), "test_R2_sd": np.std(te_r2, ddof=1) if len(te_r2) > 1 else 0})
        print(t, m, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "national" / "tuning_paradox.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
