# Code documentation

Detailed description of every script and module in `code/`. The source files themselves contain no comments; this document explains what each part does, the design decisions, the inputs and the outputs. Section and table numbers refer to the revised manuscript.

Contents
1. [Pipeline overview](#1-pipeline-overview)
2. [Shared modules](#2-shared-modules)
3. [Pipeline scripts](#3-pipeline-scripts)
4. [Leakage-control checklist](#4-leakage-control-checklist)
5. [Output files](#5-output-files)
6. [Reproducing a single result](#6-reproducing-a-single-result)

---

## 1. Pipeline overview

```
raw BMD files ──► 01_build_dataset.py ──► data/panel_monthly_24st.csv, panel_monthly_33st.csv, daily_panel.csv, qc_report.json
                                            │
             70_trends.py ◄─────────────────┤  (Mann-Kendall / Sen trends)
             20_national.py, 21_tuning_paradox.py  (re-analysis of the first-submission experiment)
             10_panel_forecast.py ──────────┤  (main experiment, ~6 h on a 2-core CPU)
                                            ▼
run_post.sh:  30_analysis.py → 50_intervals.py → 40_spatial_lodo.py → 31_diagnostics.py → 60_daily.py → 80_figures.py
                                            ▼
                         results/  (tables)       figures/  (PNG)
```

| Step | Script | Manuscript section | Runtime* |
|---|---|---|---|
| 1 | `01_build_dataset.py` | 3.1–3.2, Table 5 | ~1 min |
| 1b | `05_historical_figures.py` | 4.1, Figures 2-4, 7-41 | ~1 min |
| 2 | `70_trends.py` | 4.1.5, Table 8 | seconds |
| 3 | `20_national.py`, `21_tuning_paradox.py` | 4.2, Tables 9–10 | ~1 h |
| 4 | `10_panel_forecast.py` | 4.3, Tables 7, 11 | ~6 h |
| 5 | `30_analysis.py` | 4.3–4.5, 4.9–4.10 | ~2 min |
| 6 | `50_intervals.py` | 4.6, Table 18 | ~5 min |
| 7 | `40_spatial_lodo.py` | 4.5, Table 17 | ~25 min |
| 8 | `31_diagnostics.py` | 4.7–4.8 | ~40 min |
| 9 | `60_daily.py` | 4.11, Table 21 | ~20 min |
| 10 | `80_figures.py` | all figures | ~1 min |

\*Intel Core i5-4210H, 2 cores / 4 threads, 16 GB RAM, no GPU. Use 2 threads (`n_jobs=2`); XGBoost becomes ~30× slower when more threads are requested than physical cores.

---

## 2. Shared modules

### `common.py` – configuration, metadata, metrics
* **Paths.** `ROOT` is the repository root; `DATA`, `RESULTS` and `FIGURES` are created automatically. Raw BMD files are read from `BMD_RAW_DIR` (environment variable, default `raw_data/`). The expected raw files are:
  * `Prediction-Code/Prediction/Rainfall Prediction/(1980-2022)/All1980_2022.csv` – 25-station monthly temperature, humidity and rainfall, 1980–2022
  * `other_data.xlsx` – 29-station daily pressure, wind speed, wind direction and dew point, 1981–2022
  * `Temp,Rainfall&Humidity(1999-2022)-Latest/…/Monthly_all_Data99.csv` – 33-station monthly data, 1999–2022
  * `Prediction-Code/…/All_Station1980.csv`, `…/average_weather_data.csv` – national series used in the first submission
  * `mycodes/merged_weather_dataset_daily.csv` – daily station panel
* **Station metadata.** `STATION_ALIASES` harmonises names between archives (e.g. "Maijdee Court" → "M.Court"); `STATION_COORDS` holds the latitude/longitude of 34 stations (manuscript Table 4); `STATION_DIVISION` maps each station to one of the eight administrative divisions.
* **Evaluation design.**
  * `TEST_YEARS = 2018…2022` – outer rolling-origin test years;
  * `DEV_END = 2017` – hyperparameters are tuned on data up to this year only;
  * `INNER_VAL_BLOCKS` – the five expanding-window validation blocks 2008–09, 2010–11, 2012–13, 2014–15, 2016–17;
  * `SEEDS` (10) and `DL_SEEDS` (3) – random seeds for stochastic models and recurrent networks;
  * `SEASON_OF_MONTH` – BMD seasons: winter (DJF), pre-monsoon (MAM), monsoon (JJAS), post-monsoon (ON).
* **Metrics.** `r2()` is the coefficient of determination 1 − SS_res/SS_tot (dimensionless, Eq. 5). `metrics()` returns MAE, RMSE, bias (forecast − observation), RAE, R², Pearson r, N and, if a reference forecast is given, the MSE skill score `MSESS_clim = 1 − MSE_model / MSE_clim` (Eq. 6).
* **Helpers.** `haversine_km()` (great-circle distance), `save_json()`, and the `Timer` context manager used for computational-cost reporting.

### `features.py` – forecast predictors
A row (station *s*, target month *m*) uses only information observed up to month *m − 1*, so every predictor is available at forecast time.
* `add_lags(df, targets, dyn)`
  * sorts by station and month and builds a continuous month index `t`;
  * creates lags 1–12 of every dynamic variable, with an explicit check that the lag is exactly *k* calendar months (gaps give missing values instead of wrong lags);
  * rainfall lags are `log1p`-transformed to reduce skewness;
  * adds untransformed lag-1 and lag-12 values (`*_raw_lag1`, `*_raw_lag12`) for the persistence and seasonal-naive baselines;
  * adds the 3-month running mean of each target, the dew-point depression `DPD_lag1 = T − Td` at *t − 1*, and sine/cosine encodings of the target month;
  * keeps only rows with a complete 12-month history, so every model, including LSTM/GRU, is evaluated on identical rows.
* `neighbour_table(stations, k=4)` – the four nearest stations by haversine distance with inverse-distance-squared weights.
* `add_neighbour_lag(df, target, allowed_stations)` – IDW mean of the neighbours' **previous-month** target value. Only stations in `allowed_stations` may act as neighbours, which is used by the leave-one-division-out experiment. If no neighbour value exists, the station's own lag is used.
* `add_fold_features(df, target, train_mask)` – station × calendar-month climatology (mean, `clim`) and its standard deviation (`clim_sd`), computed from training rows only. Stations without training data fall back to the national calendar-month climatology.
* `tabular_features(target, dyn)` – the predictor list of Table 6: target lags 1, 2, 3 and 12, 3-month mean, previous-month values of the other variables, dew-point depression, month encoding, latitude, longitude, neighbour lag and climatology.
* `sequence_arrays(df, dyn)` and `static_features(target)` – inputs of the recurrent networks: a (n, 12, variables + 2) array of the previous 12 months (oldest first, including month encoding), and the static predictors (month encoding, location, neighbour lag, climatology).

### `models.py` – model zoo
* `BASELINES` = climatology, persistence, seasonal naive, SARIMA; `ML` = Ridge, DT, RF, SVR, KNN, XGB, MLP, LSTM, GRU.
* `GRIDS` and `N_CONFIGS` define the hyperparameter search spaces (manuscript Table 7); `sample_configs()` draws a fixed random subset of each grid (seed 0).
* `make_estimator(name, params, seed)`
  * every scikit-learn/XGBoost model is wrapped in a `Pipeline` with a `StandardScaler` fitted on the training data;
  * Ridge, SVR and KNN additionally standardise the target with a `TransformedTargetRegressor`;
  * fixed settings: RF 300 trees; XGBoost histogram algorithm and squared-error loss; SVR RBF kernel; `n_jobs=2`.
* `fit_predict()` – uniform interface; the PyTorch models receive the chronological validation mask for early stopping.

### `dl.py` – PyTorch networks
* `_MLP` – fully connected network with ReLU and dropout.
* `_RNN` – one LSTM or GRU layer; its last hidden state is concatenated with the static predictors and passed through a 32-unit dense head.
* `TorchRegressor` implements the training protocol shared by MLP, LSTM and GRU:
  1. inputs and target are standardised with statistics of the training window only;
  2. the last three training years form a chronological validation block;
  3. training uses Adam and mean-squared-error loss, with early stopping (patience 20) on that validation block, and the best epoch E\* is recorded;
  4. the network is re-trained on the full training window for E\* epochs, so that it sees the same data as the other models;
  5. per-epoch training (and validation) loss is stored in `history_`. With `curves=True` the full training-set R² is also recorded every epoch; this is used only for the diagnostic learning curves.
* Threads are limited to 2 (`torch.set_num_threads(2)`).

---

## 3. Pipeline scripts

### `01_build_dataset.py` – quality control, imputation, station panels
Applied identically to every station; all counts are stored in `data/qc_report.json` (manuscript Table 5).
1. **Name harmonisation** and removal of duplicated station-month records.
2. **Physical range checks.** Values outside the range are set to missing.

   | Variable | Monthly range | Daily range |
   |---|---|---|
   | Temperature | 12–36 °C | – |
   | Relative humidity | 30–100 % | – |
   | Rainfall | 0–3000 mm | – |
   | Pressure | 97–104 kPa | 94–106 kPa |
   | Wind speed | 0–60 km/h | 0–150 km/h |
   | Wind direction | – | 0–360° |
   | Dew point | −5 to 35 °C | −10 to 35 °C |

   This step also catches the **zero-coded missing values** of the original archive: 156 temperature and 133 humidity station-months.
3. **Robust outlier screen.** A value is flagged when z = 0.6745 (x − median) / MAD > 5, computed per station and calendar month, for temperature, humidity, pressure, wind speed and dew point. Rainfall is not outlier-screened, because heavy-rainfall months are genuine signal.
4. **Consistency.** A monthly dew point more than 1 °C above the monthly mean temperature is rejected.
5. **Aggregation.** Daily auxiliary variables become monthly means only when at least 20 valid days exist. Wind direction is averaged as its sine and cosine components, so that 350° and 10° average to 0°, not 180°.
6. **Temporal imputation.** Gaps of at most 2 consecutive months are filled by linear interpolation within the same station.
7. **Spatial imputation (predictors only).** Remaining gaps in pressure, wind and dew point are filled by inverse-distance-squared weighting of at least 2 stations within 100 km in the same month. Target variables are **never** spatially imputed.
8. **Final cleaning.** Rows that are still incomplete are removed. 1980 is dropped because the auxiliary record starts in 1981, and stations without an auxiliary record (Sitakund) are excluded from the unified panel.

Outputs:
* `data/panel_monthly_24st.csv` – 24 stations, 11,812 station-months, 1981–2022, all variables
* `data/panel_monthly_33st.csv` – 33 stations, 1999–2022 (temperature, humidity and rainfall only)
* `data/daily_panel.csv` – daily station panel
* `data/qc_report.json` – the counts of every QC and imputation step

### `05_historical_figures.py` – historical-analysis figures
Regenerates the exploratory figures of Section 4.1 (Figures 2–4 and 7–41) from the original BMD daily temperature and humidity records and monthly rainfall records (1999–2022: 34 stations; 1980–2022: 25 stations, 31 for rainfall).
* **Averaging (method of the first submission, verified against the published values):**
  * yearly and monthly means are taken over all station-days (temperature, humidity) or station-months (rainfall);
  * Tavg is (Tmax + Tmin) / 2;
  * temperature and humidity use the six Bengali seasons by calendar day (Summer 14 Apr–13 Jun, Monsoon 14 Jun–14 Aug, Autumn 15 Aug–13 Oct, Late Autumn 14 Oct–13 Dec, Winter 14 Dec–13 Feb, Spring 14 Feb–13 Apr);
  * rainfall uses three seasons: Summer (Mar–May), Rainy (Jun–Oct) and Winter (Nov–Feb).
* **Two versions:**
  * `original` applies the method to the unmodified files. It reproduces 84 of the 99 values printed in the published figures exactly and 12 within 0.07. The 1980–2022 rainfall figures cannot be reproduced from the current data files.
  * `qc`, used in the revised manuscript, applies daily quality control: Tmax outside 0–48 °C, Tmin outside −2 to 40 °C, zero-coded values, Tmin > Tmax, and humidity outside 5–100 % are set to missing. It also assigns Ishurdi station to Rajshahi division.
* **Outputs:**
  * `figures/historical/<version>/figNN.png`, including the individual panels of the multi-panel figures;
  * `results/historical/comparison_with_published.csv`, which lists every published value next to the reproduced and QC values.

### `10_panel_forecast.py` – main experiment
* **Task.** One-month-ahead forecasts of rainfall, temperature and humidity for every station.
* **Tuning (`tune`).**
  * For every configuration drawn from the search space, the model is trained on all years before each of the five inner validation blocks and evaluated on the block. Climatology features are recomputed per fold from that fold's training years only.
  * The configuration with the lowest pooled MSE is selected.
  * LSTM/GRU are tuned on the two most recent blocks only, to keep the compute budget manageable.
  * The out-of-sample validation predictions of the selected configuration are stored (`inner_*.csv.gz`) and later used to calibrate the conformal intervals.
* **Rolling-origin test.** For each test year Y = 2018…2022, every model is re-fitted on all years before Y and forecasts the 12 months of Y. Stochastic models (RF, XGB, MLP) use 10 seeds, LSTM/GRU 3 seeds; deterministic models use one run.
* **Baselines.**
  * *Climatology* – training-period station × month mean;
  * *Persistence* – y(t − 1);
  * *Seasonal naive* – y(t − 12);
  * *SARIMA(1,0,0)(0,1,1)₁₂* – fitted per station on the years before 2018 (rainfall as log(1 + y)). It issues one-step forecasts with fixed parameters, via Kalman filtering over the observed test months.
* Rainfall forecasts are clipped at 0.
* Results are written per target and model, so the script can be restarted and skips finished combinations.

Outputs: `results/panel/<target>__<model>.csv.gz` (predictions per station-month and seed), `inner_<target>__<model>.csv.gz`, `tuning.json` (all configurations, validation scores, selected configuration), `timing.csv` (fit + predict time).

### `20_national.py` – re-analysis of the first-submission experiment
The first submission estimated a nationally averaged monthly variable from the other two variables of the **same** month: a concurrent estimation design, not a forecast. This script reproduces that experiment and removes its problems one at a time (manuscript Table 9):
* **P0** – original protocol:
  * unscaled predictors;
  * Year sin/cos using the maximum year of the full record;
  * target min-max scaled on the full record;
  * the manual hyperparameters of the old Table 4a;
  * single split (< 2018 / 2018–2022).
* **P1** – predictors and target standardised with training statistics inside a pipeline, and a linear year index.
* **P2** – hyperparameters selected by rolling-origin validation on ≤ 2017, with 10 seeds.
* **P3** – P2 applied to the national series rebuilt from the quality-controlled panel (zero-coded values removed).

Sensitivity analyses (manuscript Figures 44–45):
* **(a) Input scaling.** No scaling, min-max or standard scaling for SVR, KNN and MLP.
* **(b) SVR surface.** The (C, γ) grid of validation MSE and test R².
* **(c) MLP surface.** The architecture × L2-penalty grid over seeds.
* **(d) Epoch-wise curves.** R² and loss of the original (100, 50) MLP with unscaled and standardised inputs. Training uses years before 2014, validation 2014–2017 and test 2018–2022; the MLP is trained with `partial_fit`.

Outputs: `results/national/national_protocols.csv`, `national_best_params.csv`, `national_sensitivity.csv`, `national_mlp_curves.csv`.

### `21_tuning_paradox.py` – why tuning appeared to reduce accuracy
For every model and target of the 1980–2022 national experiment, it compares the manual configuration with the tuned one. Each is scored on the rolling-origin **validation** error, which is available before the test period, and on the 2018–2022 **test** R². A configuration that is worse on validation but better on test could only have been found by looking at the test years. Output: `results/national/tuning_paradox.csv` (manuscript Table 10).

### `30_analysis.py` – evaluation
* **Seed handling.** Metrics are computed per seed and summarised as mean ± SD; for the RMSE a 95 % confidence interval from the t distribution is also reported.
* **Breakdowns.** Results are split by test year, season and division.
* **Extremes.** Months at or above the station's 90th and 95th training-period rainfall percentile are evaluated with MAE, RMSE, bias and relative bias, plus POD, FAR and CSI for exceedance detection.
* **Diebold–Mariano tests.**
  * *Loss differential.* For each of the 60 test months, the squared error is averaged over stations and seeds.
  * *Statistic.* DM with the Harvey–Leybourne–Newbold small-sample correction (h = 1) and a Student-t reference distribution.
  * *Comparisons.* The best ML model is tested against every competitor, and every model against climatology.
  * *Multiple testing.* p-values are adjusted with the Holm procedure.
* **Residual diagnostics.** These cover the best ML model and climatology:
  * mean, SD and skewness of the residuals;
  * the conditional-bias slope of the residual on the observation;
  * the station-averaged ACF at lags 1–12;
  * the number of stations where the Ljung–Box test at lag 12 rejects white noise.
* **Conformal prediction intervals.** They are locally scaled, season-stratified and calibrated sequentially:
  * *Non-conformity score.* The score is |y − ŷ| / σ(station, month).
  * *Calibration.* For test year Y, the (1 − α) quantile comes from the out-of-sample validation residuals (≤ 2017) plus the residuals of test years before Y, so no future information is used.
  * *Interval.* It is ŷ ± q·σ, with rainfall truncated at 0.
  * *Reported.* Coverage (PICP), mean width and the interval score are given for 80 % and 90 % intervals.

Outputs: `results/tables/*.csv` (see [Section 5](#5-output-files)).

### `31_diagnostics.py` – SHAP and learning curves
* **SHAP.** Tuned RF and XGBoost models are trained on ≤ 2017 and explained on all 2018–2022 test cases with `shap.TreeExplainer`. The script stores the mean |SHAP| per predictor, the SHAP values for beeswarm and dependence plots, and the XGBoost pairwise SHAP interaction strengths (computed on 500 cases).
* **Learning curves.** MLP, LSTM and GRU are trained for a fixed 150 epochs without early stopping (training ≤ 2014, validation 2015–2017), so the full curve is visible. The loss and R² of training, validation and test are logged every epoch for 2 seeds.

### `40_spatial_lodo.py` – leave-one-division-out validation
For each of the eight divisions, Ridge, RF and XGBoost are trained on the stations of the other seven divisions (years < 2018). The neighbour predictors are rebuilt from those stations only, and the models forecast the held-out division for 2018–2022. The same test rows are also predicted by models trained on all 24 stations ("pooled"), which quantifies the cost of never having seen a station. Output: `results/spatial/lodo.csv` (manuscript Table 17).

### `50_intervals.py` – record length versus station network
This script separates the factors that the first submission confounded. All configurations use a reduced predictor set (lags of temperature, humidity and rainfall, the only variables of the 33-station network), the same hyperparameters and the 2018–2022 test period:
* **A** – 24 stations, 1981–2022;
* **B** – the same 24 stations, 1999–2022 (effect of record length);
* **C** – 33 stations, 1999–2022;
* **C24** – the C model evaluated on the 24 common stations only (effect of network density).

Output: `results/intervals/interval_comparison.csv` (manuscript Table 18).

### `60_daily.py` – daily resolution
* **Model and task.** A separate model instance with a one-day lead time, for temperature and humidity; daily rainfall is not in the archive.
* **Predictors.**
  * lags of 1, 2, 3 and 7 days and the 7-day mean of the target;
  * previous-day values of the other variables;
  * day-of-year encoding and station coordinates;
  * a station × day-of-year climatology, smoothed with a 31-day window and computed from training years only.
* **Split.** Train 1981–2017, test 2018–2022.
* **Fixed settings.** Chosen a priori because of the size of the training set (3.3 × 10⁵ rows); the test years were not used for any choice:
  * Ridge: α = 1;
  * RF: 100 trees, depth 20, leaf 5, 50 % features, 25 % bootstrap samples;
  * XGBoost: 500 trees, depth 6, learning rate 0.05, subsampling 0.8.
* **Outputs.** `results/daily/daily_metrics.csv` (manuscript Table 21).

### `70_trends.py` – trend tests
* Mann–Kendall test with a variance correction for lag-1 serial correlation (effective sample size, Yue & Wang 2004) and Sen's slope.
* Applied to national annual means of temperature and humidity and annual totals of rainfall; to each season; and to each division.
* Only station-years with every month present are used; the national value is the mean over stations.
* Output: `results/trends/mann_kendall.csv` and `national_annual_series.csv` (manuscript Table 8, Figure 42).

### `80_figures.py` – figures
Generates every new figure of the manuscript (300 dpi PNG in `figures/`):

| Figure | Content |
|---|---|
| `fig_workflow`, `fig_rolling_origin`, `fig_station_map` | schematics and station map |
| `fig_trends` | national annual series with Sen's slope trend lines |
| `fig_national_protocols`, `fig_sensitivity`, `fig_mlp_curves_national` | re-analysis of the first-submission experiment |
| `fig_main_skill`, `fig_per_year`, `fig_division_skill` | main skill scores (bars below −1 truncated and labelled) |
| `fig_residuals`, `fig_extremes`, `fig_intervals_dhaka` | residual diagnostics, extremes, prediction intervals |
| `fig_shap_*` | SHAP importance, beeswarm (combined), dependence (coloured by the strongest interaction partner) and interactions |
| `fig_dl_curves` | learning curves of the networks |
| `fig_lodo`, `fig_daily_dhaka` | spatial hold-out and daily experiment |

Each figure function is independent and is skipped if its input file does not exist.

### `run_post.sh`
Waits until `10_panel_forecast.py` has finished, then runs steps 5–10 in order and writes logs to `results/log_*.txt`. The Python interpreter can be set with `PY=/path/to/python ./run_post.sh`.

---

## 4. Leakage-control checklist

| Possible leakage path | How it is prevented | Where |
|---|---|---|
| Concurrent predictors (same-month observations) | only lags ≥ 1 month | `features.add_lags` |
| Neighbouring stations' simultaneous values | neighbour feature uses month *t − 1* only; LODO test | `features.add_neighbour_lag`, `40_spatial_lodo.py` |
| Scaling with test statistics | `StandardScaler` inside pipelines; networks standardise with training data | `models.py`, `dl.py` |
| Climatology from test years | recomputed from training years of each fit / fold | `features.add_fold_features` |
| Hyperparameters chosen on test years | rolling-origin validation on ≤ 2017 only | `10_panel_forecast.tune` |
| Random K-fold on time series | expanding-window validation blocks and rolling-origin test | `common.INNER_VAL_BLOCKS`, `TEST_YEARS` |
| Early stopping on test data | chronological validation block = last 3 training years | `dl.TorchRegressor` |
| Interval calibration with future residuals | sequential conformal calibration (past years only) | `30_analysis.conformal` |
| Spatially imputed targets | spatial IDW only for auxiliary predictors | `01_build_dataset.spatial_idw` |

---

## 5. Output files

| File | Content | Manuscript |
|---|---|---|
| `results/tables/main_metrics.csv` | MAE, RMSE, bias, R², MSESS (mean, SD, 95 % CI) per target and model | Table 11 |
| `results/tables/per_year.csv`, `per_season.csv`, `per_division.csv` | the same metrics by test year / season / division | Fig. 47, Tables 14, 16 |
| `results/tables/dm_tests.csv` | Diebold–Mariano statistics, raw and Holm-adjusted p-values | Tables 12–13 |
| `results/tables/extremes.csv` | extreme-rainfall verification | Table 15 |
| `results/tables/residual_diagnostics.csv`, `residual_acf.csv` | residual statistics and ACF | Table 19 |
| `results/tables/intervals.csv` | conformal interval coverage and width | Table 20 |
| `results/tables/timing.csv` | training + forecast time per model | Table 22 |
| `results/tables/best_models.json` | best ML model per target | – |
| `results/panel/tuning.json` | every evaluated configuration and the selected one | Table 7 |
| `results/national/*.csv` | re-analysis of the first-submission experiment | Tables 9–10, Figs. 43–45 |
| `results/spatial/lodo.csv` | leave-one-division-out results | Table 17 |
| `results/intervals/interval_comparison.csv` | record length vs station network | Table 18 |
| `results/diagnostics/shap_importance.csv`, `shap_interactions.csv`, `dl_learning_curves.csv` | SHAP and learning curves | Figs. 51–55 |
| `results/daily/daily_metrics.csv` | daily experiment | Table 21 |
| `results/trends/mann_kendall.csv` | trend tests | Table 8 |

Station-level prediction files (`results/panel/<target>__<model>.csv.gz`, interval series, SHAP inputs) contain station observations and are therefore not distributed; they are regenerated by steps 4–8.

---

## 6. Reproducing a single result

```bash
cd code
python 10_panel_forecast.py --targets Temperature --models Climatology XGB
python 30_analysis.py
```

`--targets` and `--models` restrict the main experiment to a subset; finished target/model combinations in `results/panel/` are skipped automatically, so an interrupted run can simply be restarted.
