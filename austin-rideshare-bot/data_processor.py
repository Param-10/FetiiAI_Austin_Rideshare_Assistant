from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_EXCEL_NAME = "FetiiAI_Data_Austin.xlsx"

# Austin bounds for coordinate checks
AUSTIN_BOUNDS = {
    "lat_min": 30.10,
    "lat_max": 30.50,
    "lon_min": -98.05,
    "lon_max": -97.45,
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    for pat in patterns:
        regex = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if regex.search(c):
                return c
    return None


def _coalesce_columns(df: pd.DataFrame, targets: Dict[str, List[str]]) -> pd.DataFrame:
    """Standardize columns by first matching candidate."""
    if df is None or df.empty:
        return df
    df = df.copy()
    for new_col, candidates in targets.items():
        src = _find_column(df, candidates)
        if src and src in df.columns and new_col != src:
            df[new_col] = df[src]
        elif new_col not in df.columns:
            df[new_col] = np.nan
    return df


def _parse_datetime(series: pd.Series) -> pd.Series:
    if series is None:
        return series
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        # Convert to US Central time
        dt = dt.dt.tz_convert("America/Chicago")
    except Exception:
        # Localize to UTC if needed
        try:
            dt = dt.dt.tz_localize("UTC").dt.tz_convert("America/Chicago")
        except Exception:
            pass
    return dt


def _add_trip_features(trips: pd.DataFrame) -> pd.DataFrame:
    if trips is None or trips.empty:
        return trips
    trips = trips.copy()

    # Standardize columns
    trips = _coalesce_columns(
        trips,
        {
            "trip_id": [r"^trip[_\s-]*id$", r"^id$"],
            "booker_user_id": [r"user[_\s-]*id.*book", r"booker[_\s-]*id", r"user[_\s-]*id$"],
            "pickup_latitude": [r"pickup.*lat"],
            "pickup_longitude": [r"pickup.*(lon|lng|long)"],
            "dropoff_latitude": [r"drop[-_\s]*off.*lat", r"dropoff.*lat"],
            "dropoff_longitude": [r"drop[-_\s]*off.*(lon|lng|long)", r"dropoff.*(lon|lng|long)"],
            "pickup_address": [r"pickup.*address"],
            "dropoff_address": [r"drop[-_\s]*off.*address", r"dropoff.*address"],
            "pickup_time": [r"pickup.*(time|timestamp|date)", r"^pickup$"],
            "dropoff_time": [r"drop[-_\s]*off.*(time|timestamp|date)", r"dropoff.*(time|timestamp|date)", r"^dropoff$"],
            "num_riders": [r"(num|number|party|group).*rider|party[_\s-]*size", r"riders$"],
        },
    )

    # Timestamps
    if "pickup_time" in trips.columns:
        trips["pickup_time"] = _parse_datetime(trips["pickup_time"])
    if "dropoff_time" in trips.columns:
        trips["dropoff_time"] = _parse_datetime(trips["dropoff_time"])

    # Canonical event time
    if "event_time" not in trips.columns:
        trips["event_time"] = trips["pickup_time"].fillna(trips["dropoff_time"])

    # Derived time features
    if pd.api.types.is_datetime64_any_dtype(trips["event_time"]):
        trips["hour"] = trips["event_time"].dt.hour
        trips["day_of_week"] = trips["event_time"].dt.day_name()
        trips["date"] = trips["event_time"].dt.date
        trips["month"] = trips["event_time"].dt.month
        trips["year"] = trips["event_time"].dt.year
        trips["is_weekend"] = trips["day_of_week"].isin(["Friday", "Saturday", "Sunday"])

    # Drop out-of-bounds coordinates
    def _within_bounds(lat, lon):
        try:
            return (
                AUSTIN_BOUNDS["lat_min"] <= float(lat) <= AUSTIN_BOUNDS["lat_max"]
                and AUSTIN_BOUNDS["lon_min"] <= float(lon) <= AUSTIN_BOUNDS["lon_max"]
            )
        except Exception:
            return False

    if {"pickup_latitude", "pickup_longitude"}.issubset(trips.columns):
        mask = trips[["pickup_latitude", "pickup_longitude"]].apply(
            lambda r: _within_bounds(r["pickup_latitude"], r["pickup_longitude"]), axis=1
        )
        trips.loc[~mask, ["pickup_latitude", "pickup_longitude"]] = np.nan

    if {"dropoff_latitude", "dropoff_longitude"}.issubset(trips.columns):
        mask = trips[["dropoff_latitude", "dropoff_longitude"]].apply(
            lambda r: _within_bounds(r["dropoff_latitude"], r["dropoff_longitude"]), axis=1
        )
        trips.loc[~mask, ["dropoff_latitude", "dropoff_longitude"]] = np.nan

    return trips


def _add_demographics_features(demo: pd.DataFrame) -> pd.DataFrame:
    if demo is None or demo.empty:
        return demo
    demo = demo.copy()
    demo = _coalesce_columns(
        demo,
        {
            "user_id": [r"^user[_\s-]*id$", r"user$"],
            "age": [r"^age$", r"age[_\s-]*years?"],
        },
    )
    # Age binning
    demo["age"] = pd.to_numeric(demo["age"], errors="coerce")
    bins = [-np.inf, 17, 24, 34, 44, 54, 64, np.inf]
    labels = ["under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    demo["age_group"] = pd.cut(demo["age"], bins=bins, labels=labels)
    return demo


def _load_excel(path: Path) -> Dict[str, pd.DataFrame]:
    try:
        trips = pd.read_excel(path, sheet_name="Trip Data", engine="openpyxl")
    except Exception:
        trips = pd.DataFrame()
    try:
        riders = pd.read_excel(path, sheet_name="Rider Data", engine="openpyxl")
    except Exception:
        riders = pd.DataFrame()
    try:
        demo = pd.read_excel(path, sheet_name="Ride Demo", engine="openpyxl")
    except Exception:
        demo = pd.DataFrame()

    return {
        "trips_raw": _standardize_columns(trips),
        "riders_raw": _standardize_columns(riders),
        "demo_raw": _standardize_columns(demo),
    }


def _find_csv(pattern_contains: List[str]) -> Optional[Path]:
    # Find likely CSV file by substrings
    for p in BASE_DIR.glob("*.csv"):
        name_lower = p.name.lower()
        if all(token.lower() in name_lower for token in pattern_contains):
            return p
    return None


def _load_csvs() -> Dict[str, pd.DataFrame]:
    trips_csv = _find_csv(["trip", "data"]) or _find_csv(["trip"]) or _find_csv(["FetiiAI_Data_Austin.xlsx - Trip Data"])  # type: ignore[arg-type]
    riders_csv = _find_csv(["rider"]) or _find_csv(["passenger"]) or _find_csv(["Rider Data"])  # type: ignore[arg-type]
    demo_csv = _find_csv(["demo"]) or _find_csv(["demographic"]) or _find_csv(["Ride Demo"])  # type: ignore[arg-type]

    def _read(p: Optional[Path]) -> pd.DataFrame:
        if p and p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                # Try Excel if CSV fails
                try:
                    return pd.read_excel(p, engine="openpyxl")
                except Exception:
                    return pd.DataFrame()
        return pd.DataFrame()

    return {
        "trips_raw": _standardize_columns(_read(trips_csv)),
        "riders_raw": _standardize_columns(_read(riders_csv)),
        "demo_raw": _standardize_columns(_read(demo_csv)),
    }


def load_datasets() -> Dict[str, pd.DataFrame]:
    """Load trips, riders, and demographics (Excel or CSV)."""
    excel_path = BASE_DIR / DATA_EXCEL_NAME
    if excel_path.exists():
        raw = _load_excel(excel_path)
    else:
        raw = _load_csvs()

    trips = _add_trip_features(raw.get("trips_raw", pd.DataFrame()))

    riders = raw.get("riders_raw", pd.DataFrame())
    if not riders.empty:
        riders = _coalesce_columns(
            riders,
            {
                "trip_id": [r"^trip[_\s-]*id$"],
                "user_id": [r"^user[_\s-]*id$"],
            },
        )

    demo = _add_demographics_features(raw.get("demo_raw", pd.DataFrame()))

    meta: Dict[str, object] = {}
    if not trips.empty and pd.api.types.is_datetime64_any_dtype(trips.get("event_time")):
        meta["time_range"] = {
            "min": trips["event_time"].min(),
            "max": trips["event_time"].max(),
        }
        meta["n_trips"] = int(trips.shape[0])

    return {
        "trips": trips,
        "riders": riders,
        "demographics": demo,
        "meta": meta,
    }
