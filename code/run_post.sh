#!/bin/bash
cd "$(dirname "$0")"
PY=${PY:-python}
while pgrep -f 10_panel_forecast.py > /dev/null; do sleep 30; done
$PY -u 30_analysis.py      > ../results/log_analysis.txt 2>&1
$PY -u 50_intervals.py     > ../results/log_intervals.txt 2>&1
$PY -u 40_spatial_lodo.py  > ../results/log_lodo.txt 2>&1
$PY -u 31_diagnostics.py   > ../results/log_diagnostics.txt 2>&1
$PY -u 60_daily.py         > ../results/log_daily.txt 2>&1
$PY -u 80_figures.py       > ../results/log_figures.txt 2>&1
echo done > ../results/POST_DONE
