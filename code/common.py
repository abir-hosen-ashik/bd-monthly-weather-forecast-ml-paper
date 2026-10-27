from pathlib import Path
import json
import os
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXISTING = Path(os.environ.get("BMD_RAW_DIR", ROOT / "raw_data"))
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _d in (DATA, RESULTS, FIGURES):
    _d.mkdir(exist_ok=True)

SRC_PANEL_1980 = EXISTING / "Prediction-Code/Prediction/Rainfall Prediction/(1980-2022)/All1980_2022.csv"
SRC_OTHER_DAILY = EXISTING / "other_data.xlsx"
SRC_PANEL_1999 = EXISTING / "Temp,Rainfall&Humidity(1999-2022)-Latest/Temp,Rainfall&Humidity(1999-2022)/Monthly_all_Data99.csv"
SRC_NATIONAL_1980 = EXISTING / "Prediction-Code/Prediction/Rainfall Prediction/(1980-2022)/All_Station1980.csv"
SRC_NATIONAL_1999 = EXISTING / "Prediction-Code/Prediction/Rainfall Prediction/(1999-2022)/Bangladesh/average_weather_data.csv"
SRC_DAILY = EXISTING / "mycodes/merged_weather_dataset_daily.csv"

STATION_ALIASES = {
    "Chittagong(City - Ambagan)": "Chittagong",
    "Maijdee Court": "M.Court",
    "Sitakunda": "Sitakund",
    "Syedpur": "Sydpur",
}

STATION_COORDS = {
    "Barisal": (22.70, 90.36), "Bhola": (22.70, 90.66), "Bogra": (24.88, 89.36),
    "Chandpur": (23.26, 90.67), "Chittagong": (22.35, 91.81), "Chuadanga": (23.65, 88.81),
    "Comilla": (23.48, 91.19), "Cox's Bazar": (21.46, 91.98), "Dhaka": (23.78, 90.39),
    "Dinajpur": (25.63, 88.66), "Faridpur": (23.61, 89.84), "Feni": (23.01, 91.37),
    "Hatiya": (22.29, 91.13), "Ishurdi": (24.12, 89.04), "Jessore": (23.17, 89.22),
    "Khepupara": (21.98, 90.22), "Khulna": (22.80, 89.58), "Kutubdia": (21.83, 91.84),
    "Madaripur": (23.17, 90.18), "M.Court": (22.83, 91.08), "Mongla": (22.43, 89.66),
    "Mymensingh": (24.75, 90.41), "Patuakhali": (22.36, 90.34), "Rajshahi": (24.35, 88.56),
    "Rangamati": (22.67, 92.20), "Rangpur": (25.72, 89.26), "Sandwip": (22.50, 91.46),
    "Satkhira": (22.68, 89.07), "Sitakund": (22.64, 91.64), "Srimangal": (24.29, 91.73),
    "Sydpur": (25.75, 88.91), "Sylhet": (24.88, 91.93), "Tangail": (24.15, 89.55),
    "Teknaf": (20.87, 92.26),
}

STATION_DIVISION = {
    "Barisal": "Barisal", "Bhola": "Barisal", "Patuakhali": "Barisal", "Khepupara": "Barisal",
    "Chittagong": "Chittagong", "Comilla": "Chittagong", "Cox's Bazar": "Chittagong",
    "Feni": "Chittagong", "Hatiya": "Chittagong", "Kutubdia": "Chittagong",
    "M.Court": "Chittagong", "Rangamati": "Chittagong", "Sandwip": "Chittagong",
    "Sitakund": "Chittagong", "Teknaf": "Chittagong", "Chandpur": "Chittagong",
    "Dhaka": "Dhaka", "Faridpur": "Dhaka", "Madaripur": "Dhaka", "Tangail": "Dhaka",
    "Khulna": "Khulna", "Jessore": "Khulna", "Satkhira": "Khulna", "Mongla": "Khulna",
    "Chuadanga": "Khulna",
    "Mymensingh": "Mymensingh",
    "Rajshahi": "Rajshahi", "Bogra": "Rajshahi", "Ishurdi": "Rajshahi",
    "Rangpur": "Rangpur", "Dinajpur": "Rangpur", "Sydpur": "Rangpur",
    "Sylhet": "Sylhet", "Srimangal": "Sylhet",
}

TARGETS = ["Rainfall", "Temperature", "Humidity"]
UNITS = {"Rainfall": "mm", "Temperature": "°C", "Humidity": "%"}

TEST_YEARS = [2018, 2019, 2020, 2021, 2022]
DEV_END = 2017
INNER_VAL_BLOCKS = [(2008, 2009), (2010, 2011), (2012, 2013), (2014, 2015), (2016, 2017)]
SEEDS = list(range(10))
DL_SEEDS = list(range(3))

SEASON_OF_MONTH = {12: "Winter", 1: "Winter", 2: "Winter",
                   3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
                   6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
                   10: "Post-monsoon", 11: "Post-monsoon"}


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def metrics(y, p, ref=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    out = {
        "MAE": np.mean(np.abs(e)),
        "RMSE": np.sqrt(np.mean(e ** 2)),
        "Bias": np.mean(e),
        "RAE": np.sum(np.abs(e)) / np.sum(np.abs(y - y.mean())),
        "R2": r2(y, p),
        "r": np.corrcoef(y, p)[0, 1] if np.std(p) > 0 else np.nan,
        "N": len(y),
    }
    if ref is not None:
        ref = np.asarray(ref, float)
        out["MSESS_clim"] = 1.0 - np.mean(e ** 2) / np.mean((ref - y) ** 2)
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
