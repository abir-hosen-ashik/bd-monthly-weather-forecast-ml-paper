# Leakage-controlled machine learning forecasting of monthly rainfall, temperature and humidity across Bangladesh

Code accompanying the manuscript **"Leakage controlled machine learning forecasting of monthly rainfall temperature and humidity across Bangladesh"** (J. R. Das, M. A. R. Khan, I. Ahammad, A. Hosen; Noakhali Science and Technology University), submitted to *Discover Environment*.

The repository contains the complete, reproducible pipeline: quality control of Bangladesh Meteorological Department (BMD) station records, construction of strictly lagged forecast predictors, nested rolling-origin hyperparameter tuning, a rolling-origin test over 2018–2022, statistical evaluation, interpretability, uncertainty quantification, all figures and the result tables reported in the paper.

---

## What the study does

* **Task:** one-month-ahead forecasts of monthly total rainfall, mean temperature and mean relative humidity at 24 BMD stations (1981–2022). A separate daily experiment (one-day-ahead temperature and humidity) is included.
* **Leakage control:**
  * every predictor is observed at month *t−1* or earlier (no concurrent covariates);
  * neighbour-station predictors use only the previous month;
  * scalers, target standardisation, station climatologies and conformal quantiles are fitted on training years only;
  * hyperparameters are selected by expanding-window rolling-origin validation on ≤ 2017 (five 2-year blocks); the test years 2018–2022 are never used for any choice;
  * models are re-fitted annually over the test period (rolling origin).
* **Models (13):** climatology, persistence, seasonal naive, SARIMA, Ridge, decision tree, random forest, SVR, k-nearest neighbours, XGBoost, MLP, LSTM and GRU. Stochastic models are repeated with 10 seeds (3 for LSTM/GRU).
* **Evaluation:** MAE, RMSE, bias, R² and the MSE skill score relative to climatology, Diebold–Mariano tests (Harvey–Leybourne–Newbold correction, Holm adjustment), per year, per season, per division and for extreme rainfall (POD/FAR/CSI), plus leave-one-division-out spatial validation.
* **Diagnostics:** SHAP importance, dependence and interaction values (RF, XGBoost), epoch-wise learning curves (MLP, LSTM, GRU), residual analysis (ACF, Ljung–Box) and season-stratified conformal prediction intervals.
* **Re-analysis:** a step-by-step reproduction of the first-submission experiment explaining why hyperparameter tuning appeared to reduce accuracy (unscaled inputs, test-informed manual settings).
* **Climate trends:** Mann–Kendall tests with serial-correlation correction and Sen's slope.

### Main results (2018–2022 test period)

| Target | Best ML model (MSESS vs climatology) | Most accurate overall |
|---|---|---|
| Rainfall | SVR (+0.21) | SARIMA |
| Temperature | XGBoost (+0.32) | XGBoost |
| Humidity | GRU (+0.59) | SARIMA |

Mean temperature increased significantly by about 0.23 °C per decade (1982–2022); rainfall and humidity trends are not significant. The full tables are in `results/tables/`.

---

## Repository structure

A detailed description of every script, the design decisions, the leakage-control checklist and all output files is given in **[CODE_DETAILS.md](CODE_DETAILS.md)**.

```
code/
  common.py              paths, station metadata, metrics, evaluation design
  features.py            lagged / neighbour / climatology predictors
  models.py              model zoo, search spaces, fit/predict wrapper
  dl.py                  PyTorch MLP, LSTM, GRU with chronological early stopping
  01_build_dataset.py    quality control, imputation, station panels
  05_historical_figures.py  historical-analysis figures (Figures 2-4, 7-41) from quality-controlled daily records
  10_panel_forecast.py   main experiment (tuning + rolling-origin test)
  20_national.py         re-analysis of the first-submission experiment
  21_tuning_paradox.py   manual vs tuned configurations
  30_analysis.py         metrics, DM tests, extremes, residuals, conformal intervals
  31_diagnostics.py      SHAP and learning curves
  40_spatial_lodo.py     leave-one-division-out validation
  50_intervals.py        record length vs station network
  60_daily.py            daily-resolution experiment
  70_trends.py           Mann-Kendall / Sen trend tests
  80_figures.py          all figures
  run_post.sh            runs every step after the main experiment
  requirements.txt
results/                 result tables (CSV/JSON) used in the paper
figures/                 all figures of the paper (PNG, 300 dpi)
```

---

## Data

The quality-controlled station data sets used in the paper are published on Kaggle:
**https://www.kaggle.com/datasets/abirhosenashik/bangladesh-bmd-station-weather-data-19812022-qc**
(`panel_monthly_24st.csv`, `panel_monthly_33st.csv`, `daily_panel.csv`, `qc_report.json`). Download them into `data/` to reproduce the results from step 2 onward. The original raw station records belong to the **Bangladesh Meteorological Department (BMD)** and are available from BMD on request; they are not included here.

Raw BMD files are read from the folder given by the environment variable `BMD_RAW_DIR` (default: `raw_data/` in the repository root); the expected sub-paths are listed in `code/common.py`. Step 1 (`01_build_dataset.py`) is only needed if you have the raw BMD files.

`results/` contains only aggregated metrics and statistics (no station-level observations or predictions); running the pipeline regenerates the full prediction files in `results/panel/`.


---

## How to run

Requirements: Python 3.12 (tested on Ubuntu Linux, CPU only).

Requirements are listed in `code/requirements.txt`.

```bash
pip install -r code/requirements.txt
cd code
python 01_build_dataset.py     # step 1: QC and station panels (needs raw BMD files)
python 05_historical_figures.py  # historical-analysis figures (needs raw BMD files)
python 70_trends.py            # step 2: trend tests
python 20_national.py          # step 3: re-analysis of the first-submission experiment
python 21_tuning_paradox.py
python 10_panel_forecast.py    # step 4: main experiment (~6 h on a 2-core CPU)
./run_post.sh                  # step 5: analysis, SHAP, LODO, intervals, daily, figures (~2 h)
```

All random seeds are fixed. The results were produced on an Intel Core i5-4210H (2 cores, 4 threads), 16 GB RAM, no GPU, with scikit-learn 1.9, XGBoost 3.4, PyTorch 2.14 (CPU), statsmodels 0.15 and SHAP 0.54.

---

## Citation

If you use this code, please cite the paper (citation details will be added after publication).

## Licence

Code released under the MIT Licence. The BMD station data are not covered by this licence.

## Contact

Md. Ashikur Rahman Khan, Department of Information and Communication Engineering, Noakhali Science and Technology University — ashik@nstu.edu.bd
