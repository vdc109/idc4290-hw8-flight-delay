"""Prediction adapter for the HW5 predeparture flight-delay model."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "logistic_regression.joblib"
CATEGORICAL = ("OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "ROUTE")
VALID_START = date(2026, 1, 1)
VALID_END = date(2026, 3, 31)


@dataclass(frozen=True)
class Flight:
    flight_date: date
    carrier: str
    origin: str
    destination: str
    departure: time
    arrival: time
    elapsed_minutes: int
    distance_miles: float


@lru_cache(maxsize=1)
def load_artifact() -> dict:
    return joblib.load(MODEL_PATH)


def _hhmm(value: time) -> int:
    return value.hour * 100 + value.minute


def make_features(flight: Flight, artifact: dict | None = None) -> pd.DataFrame:
    """Match HW5 feature engineering and category capping exactly."""
    artifact = artifact or load_artifact()
    if not VALID_START <= flight.flight_date <= VALID_END:
        raise ValueError("Choose a date from January through March 2026.")
    if flight.origin == flight.destination:
        raise ValueError("Origin and destination must differ.")
    if flight.elapsed_minutes <= 0 or flight.distance_miles <= 0:
        raise ValueError("Duration and distance must be positive.")
    vocab = artifact["vocabulary"]
    if flight.carrier not in vocab["OP_UNIQUE_CARRIER"]:
        raise ValueError("Choose one of the airlines in the training data.")
    airports = set(vocab["ORIGIN"]) | set(vocab["DEST"])
    if flight.origin not in airports or flight.destination not in airports:
        raise ValueError("Choose airports from the available list.")

    d = pd.DataFrame([{
        "DATE": pd.Timestamp(flight.flight_date),
        "OP_UNIQUE_CARRIER": flight.carrier,
        "ORIGIN": flight.origin,
        "DEST": flight.destination,
        "CRS_DEP_TIME": _hhmm(flight.departure),
        "CRS_ARR_TIME": _hhmm(flight.arrival),
        "CRS_ELAPSED_TIME": float(flight.elapsed_minutes),
        "DISTANCE": float(flight.distance_miles),
    }])
    d["MONTH"] = d.DATE.dt.month.astype("int8")
    d["DAY_OF_WEEK"] = d.DATE.dt.dayofweek.astype("int8")
    d["DAY_OF_MONTH"] = d.DATE.dt.day.astype("int8")
    d["IS_WEEKEND"] = (d.DAY_OF_WEEK >= 5).astype("int8")
    d["DEP_HOUR"] = (d.CRS_DEP_TIME // 100).astype("int8")
    d["ARR_HOUR"] = (d.CRS_ARR_TIME // 100).astype("int8")
    dep_min = d.DEP_HOUR * 60 + d.CRS_DEP_TIME % 100
    arr_min = d.ARR_HOUR * 60 + d.CRS_ARR_TIME % 100
    d["DEP_SIN"] = np.sin(2 * np.pi * dep_min / 1440).astype("float32")
    d["DEP_COS"] = np.cos(2 * np.pi * dep_min / 1440).astype("float32")
    d["ARR_SIN"] = np.sin(2 * np.pi * arr_min / 1440).astype("float32")
    d["ARR_COS"] = np.cos(2 * np.pi * arr_min / 1440).astype("float32")
    d["ROUTE"] = d.ORIGIN + "-" + d.DEST

    x = d[artifact["features"]].copy()
    for col in CATEGORICAL:
        categories = list(vocab[col]) + ["OTHER"]
        x[col] = x[col].where(x[col].isin(vocab[col]), "OTHER").astype("category")
        x[col] = x[col].cat.set_categories(categories)
    return x


def predict_delay(flight: Flight) -> float:
    artifact = load_artifact()
    return float(artifact["model"].predict_proba(make_features(flight, artifact))[0, 1])
