import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from common import DATA, FIGURES, RESULTS, STATION_COORDS, STATION_DIVISION, TARGETS, UNITS

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
TAB = RESULTS / "tables"
COL = {"Climatology": "#9e9e9e", "Persistence": "#bdbdbd", "Seasonal naive": "#d6d6d6", "SARIMA": "#7f7f7f",
       "Ridge": "#8c6d31", "DT": "#e7969c", "RF": "#2b8cbe", "SVR": "#fdae6b", "KNN": "#c7e9c0",
       "XGB": "#08519c", "MLP": "#9467bd", "LSTM": "#d62728", "GRU": "#ff9896"}
ORDER = list(COL)


def save(fig, name):
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print("saved", name)


def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axis("off")
    boxes = [
        (0.02, 0.62, "BMD station records\n1980-2022\n(T, RH, rainfall, P,\nwind, dew point)"),
        (0.22, 0.62, "Quality control\nrange / MAD / consistency\n<=2-month interpolation\nIDW (predictors only)"),
        (0.42, 0.62, "Unified panel\n24 stations, 1981-2022\n+ 33-station panel\n(1999-2022)"),
        (0.62, 0.62, "Forecast features (t-1)\nlags 1-3, 12; roll-3\nneighbour lag (IDW)\nclimatology (train only)"),
        (0.82, 0.62, "Models\nbaselines, SARIMA\nRidge, DT, RF, SVR, KNN\nXGB, MLP, LSTM, GRU"),
        (0.82, 0.12, "Tuning: rolling-origin CV\non <=2017 only\n(5 expanding 2-yr blocks)"),
        (0.62, 0.12, "Rolling-origin test\n2018-2022, annual refit\n10 seeds (3 for RNNs)"),
        (0.42, 0.12, "Evaluation\nMAE, RMSE, bias, R², MSESS\nDM tests, extremes,\nseasons, divisions"),
        (0.22, 0.12, "Diagnostics\nSHAP, learning curves\nresiduals, LODO,\nconformal intervals"),
        (0.02, 0.12, "Outputs\nmodel ranking with\nuncertainty;\noperational guidance"),
    ]
    for x, y, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.17, 0.28, boxstyle="round,pad=0.01", fc="#eef3f8", ec="#2b5c8a", lw=0.8))
        ax.text(x + 0.085, y + 0.14, t, ha="center", va="center", fontsize=5.6)
    arrow = dict(arrowstyle="->", color="#2b5c8a", lw=0.9)
    for i in range(4):
        ax.annotate("", xy=(0.215 + 0.2 * i, 0.76), xytext=(0.195 + 0.2 * i, 0.76), arrowprops=arrow)
    ax.annotate("", xy=(0.905, 0.41), xytext=(0.905, 0.61), arrowprops=arrow)
    for i in range(4):
        ax.annotate("", xy=(0.195 + 0.2 * (3 - i), 0.26), xytext=(0.215 + 0.2 * (3 - i), 0.26), arrowprops=arrow)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "fig_workflow.png")


def fig_rolling_origin():
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    years = np.arange(1982, 2023)
    rows = []
    from common import INNER_VAL_BLOCKS
    for a, b in INNER_VAL_BLOCKS:
        rows.append(("Tuning fold", a, b))
    for Y in range(2018, 2023):
        rows.append(("Test origin", Y, Y))
    for i, (kind, a, b) in enumerate(rows):
        yv = len(rows) - i
        ax.barh(yv, a - years[0], left=years[0], color="#9ecae1", height=0.6)
        ax.barh(yv, b - a + 1, left=a, color="#fd8d3c" if kind == "Tuning fold" else "#de2d26", height=0.6)
        lab = f"inner fold {i + 1}: validate {a}-{b}" if kind == "Tuning fold" else f"outer origin: test {a}"
        ax.text(1981.3, yv, lab, ha="right", va="center", fontsize=6.5)
    ax.axvline(2018, color="k", lw=0.8, ls="--")
    ax.text(2018.2, len(rows) + 0.55, "test period", fontsize=7)
    ax.text(2017.8, len(rows) + 0.55, "development (tuning)", fontsize=7, ha="right")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(1968, 2023.5)
    ax.set_xticks([1982, 1990, 2000, 2010, 2018, 2023])
    ax.set_xlabel("Year (bars span calendar years)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#9ecae1", label="training window (expanding)"),
                       Patch(color="#fd8d3c", label="validation block (tuning)"),
                       Patch(color="#de2d26", label="test year (model re-fitted annually)")],
              fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    save(fig, "fig_rolling_origin.png")


def fig_station_map():
    p = pd.read_csv(DATA / "panel_monthly_24st.csv")
    p33 = pd.read_csv(DATA / "panel_monthly_33st.csv")
    fig, ax = plt.subplots(figsize=(4.2, 5))
    divs = sorted(set(STATION_DIVISION.values()))
    cmap = plt.get_cmap("tab10")
    for s in p33.Station.unique():
        la, lo = STATION_COORDS[s]
        in24 = s in set(p.Station)
        ax.scatter(lo, la, s=38 if in24 else 22, marker="o" if in24 else "^",
                   color=cmap(divs.index(STATION_DIVISION[s])), ec="k", lw=0.4, zorder=3)
        ax.text(lo + 0.05, la + 0.04, s, fontsize=5.5)
    for i, dv in enumerate(divs):
        ax.scatter([], [], color=cmap(i), label=dv, s=25)
    ax.scatter([], [], marker="o", color="w", ec="k", label="in 24-station panel")
    ax.scatter([], [], marker="^", color="w", ec="k", label="33-station panel only")
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect(1.05)
    save(fig, "fig_station_map.png")


def fig_main_skill():
    m = pd.read_csv(TAB / "main_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=False)
    for ax, t in zip(axes, TARGETS):
        g = m[(m.Target == t) & (m.Model != "Climatology")].copy()
        g["Model"] = pd.Categorical(g.Model, ORDER, ordered=True)
        g = g.sort_values("Model")
        ax.bar(range(len(g)), g.MSESS_clim, yerr=g.MSESS_clim_sd, color=[COL[x] for x in g.Model], capsize=2, ec="k", lw=0.3)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([m.replace("Seasonal naive", "Seas. naive") for m in g.Model], rotation=90, fontsize=7)
        ax.set_title(t)
        lo = max(-1.0, min(-0.3, g.MSESS_clim.min() - 0.05))
        ax.set_ylim(lo, max(0.5, g.MSESS_clim.max() + 0.1))
        for i, v in enumerate(g.MSESS_clim):
            if v < lo:
                ax.text(i, lo + 0.03, f"{v:.1f}", ha="center", va="bottom", fontsize=6, rotation=90, color="w")
    axes[0].set_ylabel("MSE skill score vs climatology\n(mean ± SD over seeds)")
    save(fig, "fig_main_skill.png")


def fig_per_year():
    m = pd.read_csv(TAB / "per_year.csv")
    best = json.loads((TAB / "best_models.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for ax, t in zip(axes, TARGETS):
        for mdl in ["Climatology", "SARIMA", "Ridge", "RF", "XGB", "LSTM", best[t]]:
            g = m[(m.Target == t) & (m.Model == mdl)]
            if len(g):
                ax.errorbar(g.Year, g.RMSE, yerr=g.RMSE_sd, label=mdl, color=COL[mdl], marker="o", ms=3, lw=1, capsize=2)
        ax.set_title(t)
        ax.set_xticks(range(2018, 2023))
        ax.set_ylabel(f"RMSE ({UNITS[t]})")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), ncol=7, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    save(fig, "fig_per_year.png")


def fig_division_heatmap():
    m = pd.read_csv(TAB / "per_division.csv")
    models = ["Persistence", "SARIMA", "Ridge", "DT", "RF", "SVR", "KNN", "XGB", "MLP", "LSTM", "GRU"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2), sharey=True)
    for ax, t in zip(axes, TARGETS):
        g = m[m.Target == t].pivot(index="Division", columns="Model", values="MSESS_clim").reindex(columns=models)
        im = ax.imshow(g.to_numpy(), cmap="RdBu", vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=75, fontsize=6.5)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.index, fontsize=7)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.8)
        ax.set_title(t)
    fig.colorbar(im, ax=axes, shrink=0.8, label="MSESS vs climatology")
    save(fig, "fig_division_skill.png")


def fig_national_protocols():
    n = pd.read_csv(RESULTS / "national" / "national_protocols.csv")
    n = n[n.Series.str.startswith("1980") | n.Series.str.startswith("1981")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=True)
    protos = ["P0 original", "P1 scaled, manual", "P2 scaled, tuned", "P3 QC series, scaled, tuned"]
    colors = ["#bdbdbd", "#9ecae1", "#3182bd", "#08519c"]
    models = ["DT", "RF", "SVR", "KNN", "MLP", "XGB"]
    for ax, t in zip(axes, TARGETS):
        for k, (pr, c) in enumerate(zip(protos, colors)):
            g = n[(n.Target == t) & (n.Protocol == pr)].groupby("Model").test_R2.agg(["mean", "std"]).reindex(models)
            ax.bar(np.arange(len(models)) + (k - 1.5) * 0.2, g["mean"], 0.2, yerr=g["std"].fillna(0), color=c, label=pr, capsize=1.5, ec="k", lw=0.2)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=7)
        ax.set_title(t)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-0.5, 1.0)
    axes[0].set_ylabel("Test R² (2018-2022)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    save(fig, "fig_national_protocols.png")


def fig_sensitivity():
    s = pd.read_csv(RESULTS / "national" / "national_sensitivity.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    sc = s[s.analysis == "scaling"]
    ax = axes[0]
    labs, data = [], []
    for t in ["Temperature", "Rainfall"]:
        for mdl in ["SVR", "KNN", "MLP"]:
            for scl in ["none", "minmax", "standard"]:
                v = sc[(sc.Target == t) & (sc.Model == mdl) & (sc.scaling == scl)].test_R2
                data.append(v.to_numpy())
                labs.append(f"{t[0]}-{mdl}-{scl[:3]}")
    ax.boxplot(data, widths=0.6, showfliers=False)
    ax.set_xticks(range(1, len(labs) + 1))
    ax.set_xticklabels(labs, rotation=90, fontsize=5)
    ax.set_ylabel("Test R²")
    ax.set_title("(a) input scaling", fontsize=8)
    ax.set_ylim(-1, 1)
    sv = s[(s.analysis == "svr_surface") & (s.Target == "Temperature")]
    piv = sv.pivot(index="C", columns="gamma", values="test_R2")
    im = axes[1].imshow(piv.to_numpy(), cmap="viridis", vmin=-0.5, vmax=1, origin="lower")
    axes[1].set_xticks(range(piv.shape[1]))
    axes[1].set_xticklabels(piv.columns, fontsize=6)
    axes[1].set_yticks(range(piv.shape[0]))
    axes[1].set_yticklabels(piv.index, fontsize=6)
    axes[1].set_xlabel("gamma")
    axes[1].set_ylabel("C")
    axes[1].set_title("(b) SVR, temperature: test R²", fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[1].text(j, i, f"{piv.iat[i, j]:.2f}", ha="center", va="center", fontsize=5, color="w")
    ml = s[(s.analysis == "mlp_surface") & (s.Target == "Temperature")]
    pm = ml.groupby(["hidden", "alpha"]).test_R2.mean().unstack()
    im2 = axes[2].imshow(pm.to_numpy(), cmap="viridis", vmin=-0.5, vmax=1, origin="lower")
    axes[2].set_xticks(range(pm.shape[1]))
    axes[2].set_xticklabels(pm.columns, fontsize=6)
    axes[2].set_yticks(range(pm.shape[0]))
    axes[2].set_yticklabels(pm.index, fontsize=6)
    axes[2].set_xlabel("alpha (L2)")
    axes[2].set_title("(c) MLP, temperature:\nmean test R² over 3 seeds", fontsize=8)
    for i in range(pm.shape[0]):
        for j in range(pm.shape[1]):
            axes[2].text(j, i, f"{pm.iat[i, j]:.2f}", ha="center", va="center", fontsize=5, color="w")
    save(fig, "fig_sensitivity.png")


def fig_mlp_curves_national():
    c = pd.read_csv(RESULTS / "national" / "national_mlp_curves.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    for j, t in enumerate(["Temperature", "Rainfall"]):
        for i, v in enumerate(["original (unscaled X)", "standardised X"]):
            ax = axes[i, j]
            g = c[(c.Target == t) & (c.variant == v)]
            for nm, col in [("train", "#2b8cbe"), ("val", "#fd8d3c"), ("test", "#de2d26")]:
                agg = g.groupby("epoch")[f"{nm}_R2"].agg(["mean", "min", "max"])
                ax.plot(agg.index, agg["mean"], color=col, label=nm, lw=1)
                ax.fill_between(agg.index, agg["min"], agg["max"], color=col, alpha=0.15)
            ax.set_ylim(-1, 1)
            ax.set_title(f"{t} - {v}", fontsize=8)
            ax.set_ylabel("R²")
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    axes[0, 0].legend(fontsize=7, frameon=False)
    save(fig, "fig_mlp_curves_national.png")


def fig_dl_curves():
    p = RESULTS / "diagnostics" / "dl_learning_curves.csv"
    if not p.exists():
        return
    c = pd.read_csv(p)
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.2), sharex=True)
    for i, t in enumerate(TARGETS):
        for j, mdl in enumerate(["MLP", "LSTM", "GRU"]):
            ax = axes[i, j]
            g = c[(c.Target == t) & (c.Model == mdl)]
            for nm, col in [("train", "#2b8cbe"), ("val", "#fd8d3c"), ("test", "#de2d26")]:
                agg = g.groupby("epoch")[f"{nm}_loss"].agg(["mean", "min", "max"])
                ax.plot(agg.index, agg["mean"], color=col, lw=1, label=f"{nm} loss")
                ax.fill_between(agg.index, agg["min"], agg["max"], color=col, alpha=0.15)
            vb = g.groupby("epoch")["val_loss"].mean()
            ax.axvline(vb.idxmin(), color="k", ls=":", lw=0.8)
            ax.set_title(f"{t} - {mdl}", fontsize=8)
            ax.set_ylim(0, max(1.2, float(g.val_loss.quantile(0.98))))
            if j == 0:
                ax.set_ylabel("MSE (standardised)")
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    axes[0, 0].legend(fontsize=6, frameon=False)
    save(fig, "fig_dl_curves.png")


PRETTY = {"clim": "climatology", "Month_sin": "month sin", "Month_cos": "month cos", "DPD_lag1": "dew-point depr. (t-1)"}


def pretty(f):
    if f in PRETTY:
        return PRETTY[f]
    return f.replace("_lag", " lag").replace("_roll3", " 3-mo mean").replace("nb_", "neighbour ").replace("_", " ")


def fig_shap():
    import shap
    d = RESULTS / "diagnostics"
    if not (d / "shap_importance.csv").exists():
        return
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.8), gridspec_kw={"wspace": 0.95})
    imp = pd.read_csv(d / "shap_importance.csv")
    for ax, t in zip(axes, TARGETS):
        g = imp[imp.Target == t].pivot(index="feature", columns="Model", values="mean_abs_shap")
        g = g.sort_values("XGB").tail(10)
        y = np.arange(len(g))
        ax.barh(y - 0.2, g["RF"], 0.4, color=COL["RF"], label="RF")
        ax.barh(y + 0.2, g["XGB"], 0.4, color=COL["XGB"], label="XGB")
        ax.set_yticks(y)
        ax.set_yticklabels([pretty(f) for f in g.index], fontsize=6.5)
        ax.set_xlabel(f"mean |SHAP| ({UNITS[t]})")
        ax.set_title(t)
    axes[0].legend(fontsize=7, frameon=False)
    save(fig, "fig_shap_importance.png")
    for t in TARGETS:
        sv = np.load(d / f"shap_{t}_XGB.npy")
        X = pd.read_csv(d / f"shap_X_{t}_XGB.csv")
        X.columns = [pretty(c) for c in X.columns]
        plt.figure()
        shap.summary_plot(sv, X, show=False, max_display=12, plot_size=(5.5, 4))
        plt.title(f"{t} - XGB", fontsize=9)
        plt.savefig(FIGURES / f"fig_shap_beeswarm_{t}.png")
        plt.close("all")
    from PIL import Image
    ims = [Image.open(FIGURES / f"fig_shap_beeswarm_{t}.png").convert("RGB") for t in TARGETS]
    h = max(i.height for i in ims)
    ims = [i.resize((int(i.width * h / i.height), h)) for i in ims]
    canvas = Image.new("RGB", (sum(i.width for i in ims), h), "white")
    x = 0
    for i in ims:
        canvas.paste(i, (x, 0))
        x += i.width
    canvas.save(FIGURES / "fig_shap_beeswarm.png", dpi=(300, 300))
    inter = pd.read_csv(d / "shap_interactions.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 4.8))
    for j, t in enumerate(TARGETS):
        sv = np.load(d / f"shap_{t}_XGB.npy")
        X = pd.read_csv(d / f"shap_X_{t}_XGB.csv")
        order = np.argsort(-np.abs(sv).mean(0))[:2]
        for i, k in enumerate(order):
            f = X.columns[k]
            it = inter[(inter.Target == t) & ((inter.f1 == f) | (inter.f2 == f))].iloc[0]
            partner = it.f2 if it.f1 == f else it.f1
            ax = axes[i, j]
            scat = ax.scatter(X[f], sv[:, k], c=X[partner], s=4, cmap="coolwarm")
            ax.set_xlabel(pretty(f), fontsize=7)
            ax.set_ylabel(f"SHAP ({UNITS[t]})", fontsize=7)
            cb = fig.colorbar(scat, ax=ax, pad=0.01)
            cb.set_label(pretty(partner), fontsize=6)
            cb.ax.tick_params(labelsize=5)
            ax.set_title(t, fontsize=8)
    save(fig, "fig_shap_dependence.png")
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.8))
    for ax, t in zip(axes, TARGETS):
        g = inter[inter.Target == t].head(8)
        ax.barh(range(len(g))[::-1], g.mean_abs_interaction, color="#6baed6")
        ax.set_yticks(range(len(g))[::-1])
        ax.set_yticklabels([f"{pretty(a)} × {pretty(b)}" for a, b in zip(g.f1, g.f2)], fontsize=5.5)
        ax.set_title(t, fontsize=8)
        ax.set_xlabel("mean |interaction SHAP|", fontsize=7)
    save(fig, "fig_shap_interactions.png")


def fig_residuals():
    best = json.loads((TAB / "best_models.json").read_text())
    df = pd.concat([pd.read_csv(f) for f in (RESULTS / "panel").glob("*__*.csv.gz") if not f.name.startswith("inner_")])
    acf = pd.read_csv(TAB / "residual_acf.csv")
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.6))
    for i, t in enumerate(TARGETS):
        g = df[(df.Target == t) & (df.Model == best[t])].groupby(["Station", "Year", "Month"]).agg(y=("y", "first"), pred=("pred", "mean")).reset_index()
        e = g.pred - g.y
        axes[i, 0].hist(e, bins=50, color=COL.get(best[t], "#888"), alpha=0.8)
        axes[i, 0].set_xlabel(f"residual ({UNITS[t]})")
        axes[i, 0].set_title(f"{t} - {best[t]}", fontsize=8)
        axes[i, 1].scatter(g.y, e, s=2, alpha=0.4, color="#444")
        axes[i, 1].axhline(0, color="r", lw=0.7)
        axes[i, 1].set_xlabel(f"observed ({UNITS[t]})")
        axes[i, 1].set_ylabel("residual")
        a = acf[(acf.Target == t)]
        for mdl, ls in [(best[t], "-"), ("Climatology", "--")]:
            h = a[a.Model == mdl]
            axes[i, 2].plot(h.lag, h.acf, ls, marker="o", ms=3, label=mdl)
        axes[i, 2].axhline(0, color="k", lw=0.5)
        axes[i, 2].axhspan(-1.96 / np.sqrt(60), 1.96 / np.sqrt(60), color="grey", alpha=0.2)
        axes[i, 2].set_xlabel("lag (months)")
        axes[i, 2].set_ylabel("station-mean ACF")
        axes[i, 2].legend(fontsize=6, frameon=False)
    save(fig, "fig_residuals.png")


def fig_extremes():
    ex = pd.read_csv(TAB / "extremes.csv")
    ex = ex[ex["quantile"] == 0.95]
    models = [m for m in ORDER if m in set(ex.Model)]
    ex = ex.set_index("Model").reindex(models)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    axes[0].bar(range(len(ex)), ex["RelBias_%"], color=[COL[m] for m in ex.index], ec="k", lw=0.3)
    axes[0].set_xticks(range(len(ex)))
    axes[0].set_xticklabels(ex.index, rotation=70, fontsize=7)
    axes[0].set_ylabel("relative bias on extreme months (%)")
    axes[0].axhline(0, color="k", lw=0.5)
    x = np.arange(len(ex))
    axes[1].bar(x - 0.2, ex.POD, 0.4, label="POD", color="#3182bd")
    axes[1].bar(x + 0.2, ex.CSI, 0.4, label="CSI", color="#fd8d3c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ex.index, rotation=70, fontsize=7)
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].set_title("exceedance of station 95th percentile", fontsize=8)
    save(fig, "fig_extremes.png")


def fig_intervals():
    s = pd.read_csv(TAB / "intervals_series.csv.gz")
    best = json.loads((TAB / "best_models.json").read_text())
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6), sharex=True)
    for ax, t in zip(axes, TARGETS):
        g = s[(s.Target == t) & (s.Model == best[t]) & (s.Station == "Dhaka")].sort_values(["Year", "Month"])
        x = g.Year + (g.Month - 0.5) / 12
        ax.fill_between(x, g.lo, g.hi, color=COL.get(best[t], "#888"), alpha=0.25, label="90 % conformal interval")
        ax.plot(x, g.pred, color=COL.get(best[t], "#333"), lw=1, label=f"{best[t]} forecast")
        ax.plot(x, g.y, "k.", ms=3, label="observed")
        ax.set_ylabel(f"{t} ({UNITS[t]})")
    axes[0].legend(fontsize=7, frameon=False, ncol=3)
    axes[-1].set_xlabel("Dhaka station, 2018-2022")
    save(fig, "fig_intervals_dhaka.png")


def fig_lodo():
    p = RESULTS / "spatial" / "lodo.csv"
    if not p.exists():
        return
    l = pd.read_csv(p)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=False)
    for ax, t in zip(axes, TARGETS):
        g = l[(l.Target == t) & (l.Model == "XGB")].groupby(["Division", "Setting"]).MSESS_clim.mean().unstack()
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["Pooled"], 0.4, label="pooled (station seen in training)", color="#9ecae1")
        ax.bar(x + 0.2, g["LODO"], 0.4, label="leave-one-division-out", color="#08519c")
        ax.set_xticks(x)
        ax.set_xticklabels(g.index, rotation=60, fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{t} - XGB", fontsize=8)
    axes[0].set_ylabel("MSESS vs climatology")
    axes[0].legend(fontsize=6, frameon=False)
    save(fig, "fig_lodo.png")


def fig_trends():
    nat = pd.read_csv(RESULTS / "trends" / "national_annual_series.csv")
    mk = pd.read_csv(RESULTS / "trends" / "mann_kendall.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.4))
    for ax, v in zip(axes, ["Temperature", "Humidity", "Rainfall"]):
        r = mk[(mk.Scope == "Bangladesh") & (mk.Period == "Annual") & (mk.Variable == v)].iloc[0]
        ax.plot(nat.Year, nat[v], "o-", ms=2.5, lw=0.8, color="#3182bd")
        sl = r.Sen_slope_per_decade / 10
        mid = nat[v].median()
        xm = nat.Year.median()
        ax.plot(nat.Year, mid + sl * (nat.Year - xm), "r--", lw=1)
        unit = "mm yr⁻¹" if v == "Rainfall" else UNITS[v]
        ax.set_title(f"{v}: {r.Sen_slope_per_decade:+.2f} {UNITS[v] if v != 'Rainfall' else 'mm'}/decade, p={r.p:.3f}", fontsize=7)
        ax.set_ylabel(f"annual {'total' if v == 'Rainfall' else 'mean'} ({unit if v != 'Rainfall' else 'mm'})", fontsize=7)
    save(fig, "fig_trends.png")


def fig_daily():
    p = RESULTS / "daily" / "daily_predictions.csv.gz"
    if not p.exists():
        return
    d = pd.read_csv(p, parse_dates=["Date"])
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4), sharex=True)
    for ax, t in zip(axes, ["Temperature", "Humidity"]):
        g = d[(d.Target == t) & (d.Model == "XGB") & (d.Station == "Dhaka") & (d.Date.dt.year == 2022)].sort_values("Date")
        ax.plot(g.Date, g.y, "k-", lw=0.7, label="observed")
        ax.plot(g.Date, g.pred, color=COL["XGB"], lw=0.7, label="XGB, 1-day ahead")
        ax.set_ylabel(f"{t} ({UNITS[t]})")
    axes[0].legend(fontsize=7, frameon=False)
    axes[-1].set_xlabel("Dhaka, 2022")
    save(fig, "fig_daily_dhaka.png")


if __name__ == "__main__":
    for f in [fig_workflow, fig_rolling_origin, fig_station_map, fig_trends, fig_national_protocols, fig_sensitivity,
              fig_mlp_curves_national, fig_main_skill, fig_per_year, fig_division_heatmap, fig_residuals,
              fig_extremes, fig_intervals, fig_shap, fig_dl_curves, fig_lodo, fig_daily]:
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
