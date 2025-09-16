from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from visualizations import make_map_html, plot_hourly_pattern, plot_time_series, plot_top_locations


@dataclass
class Place:
    name: str
    lat: float
    lon: float
    radius_km: float


KNOWN_PLACES = {
    "moody center": Place("Moody Center", 30.283, -97.732, 0.6),
    "downtown": Place("Downtown Austin", 30.2686, -97.742, 1.5),
    "domain": Place("The Domain", 30.4021, -97.7259, 1.0),
    "airport": Place("Austin-Bergstrom International Airport", 30.1975, -97.6664, 1.2),
    "ut": Place("UT Austin", 30.2849, -97.7341, 0.8),
    "sixth street": Place("Sixth Street", 30.2673, -97.7403, 0.5),
    "6th street": Place("Sixth Street", 30.2673, -97.7403, 0.5),
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _within_place(df: pd.DataFrame, place: Place, lat_col: str, lon_col: str) -> pd.Series:
    if df.empty or not {lat_col, lon_col}.issubset(df.columns):
        return pd.Series([False] * len(df), index=df.index)
    lat = df[lat_col].astype(float)
    lon = df[lon_col].astype(float)
    return pd.Series(
        _haversine_km(lat, lon, place.lat, place.lon) <= place.radius_km,
        index=df.index,
    )


def _filter_by_entities(trips: pd.DataFrame, entities: Dict[str, Any]) -> pd.DataFrame:
    df = trips.copy()
    if df.empty:
        return df

    # Time filters
    if "days" in entities and "day_of_week" in df.columns:
        df = df[df["day_of_week"].isin(entities["days"])].copy()

    if "hour_range" in entities and "hour" in df.columns:
        start, end = entities["hour_range"]
        if start <= end:
            df = df[(df["hour"] >= start) & (df["hour"] <= end)]
        else:
            # Handle wrap-around (20→03)
            df = df[(df["hour"] >= start) | (df["hour"] <= end)]

    # Relative time filters
    if "relative_time" in entities and "event_time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["event_time"]):
        max_time = df["event_time"].max()
        if entities["relative_time"] == "last_month":
            last_month = (max_time - pd.offsets.MonthBegin(1)).to_period("M")
            df = df[df["event_time"].dt.to_period("M") == last_month]
        elif entities["relative_time"] == "this_month":
            this_month = max_time.to_period("M")
            df = df[df["event_time"].dt.to_period("M") == this_month]
        elif entities["relative_time"] == "last_week":
            week_start = (max_time.normalize() - pd.Timedelta(days=7))
            df = df[(df["event_time"] >= week_start) & (df["event_time"] <= max_time)]

    # Group size
    size_col = None
    for c in ["num_riders", "group_size", "riders", "party_size"]:
        if c in df.columns:
            size_col = c
            break
    if size_col:
        if "group_size_min" in entities:
            df = df[df[size_col] >= entities["group_size_min"]]
        if "group_size_eq" in entities:
            df = df[df[size_col] == entities["group_size_eq"]]

    return df


def _join_with_demographics(trips: pd.DataFrame, riders: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    if trips.empty:
        return trips
    df = trips.copy()

    # Join riders to demographics by user_id
    if not riders.empty and not demo.empty and {"trip_id", "user_id"}.issubset(riders.columns) and {"user_id"}.issubset(demo.columns):
        riders_join = riders.merge(demo[["user_id", "age", "age_group"]], on="user_id", how="left")
        df = df.merge(riders_join, on="trip_id", how="left")
        return df

    # Fallback: use booker user id
    if "booker_user_id" in df.columns and not demo.empty and "user_id" in demo.columns:
        return df.merge(demo[["user_id", "age", "age_group"]], left_on="booker_user_id", right_on="user_id", how="left")

    return df


def _summarize_count(df: pd.DataFrame, label: str) -> str:
    return f"{label}: {len(df):,} trips"


def answer_query(
    trips: pd.DataFrame,
    riders: pd.DataFrame,
    demographics: pd.DataFrame,
    intent: str,
    entities: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Any]]:
    """Return (context, figures)."""
    if trips is None or trips.empty:
        return ({"summary": "I couldn't find trip data.", "highlights": []}, [])

    figures: List[Any] = []
    context: Dict[str, Any] = {"stats": {}}

    # Pre-filter by entities
    filtered = _filter_by_entities(trips, entities)

    if intent == "location":
        place_key = entities.get("place")
        loc_df = filtered
        if place_key and place_key in KNOWN_PLACES:
            place = KNOWN_PLACES[place_key]
            mask = _within_place(loc_df, place, "dropoff_latitude", "dropoff_longitude")
            loc_df = loc_df[mask].copy()
            title = f"Trips near {place.name}"
        else:
            title = "Popular Drop-offs"

        summary = _summarize_count(loc_df, label=title)
        context["summary"] = summary
        context.setdefault("highlights", []).append(summary)

        # Bar chart
        bar = plot_top_locations(loc_df)
        figures.append(bar)

        # Map
        mhtml = make_map_html(loc_df)
        if mhtml:
            figures.append(mhtml)

    elif intent == "temporal":
        tdf = filtered
        summary = _summarize_count(tdf, label="Filtered trips")
        context["summary"] = "Here's how demand changes over time."
        context.setdefault("highlights", []).append(summary)

        ts_fig = plot_time_series(tdf)
        hr_fig = plot_hourly_pattern(tdf)
        figures.extend([ts_fig, hr_fig])

    elif intent == "demographic":
        jdf = _join_with_demographics(filtered, riders, demographics)
        if not jdf.empty and "age_group" in jdf.columns:
            counts = (
                jdf["age_group"].value_counts().sort_index().reset_index()
            )
            counts.columns = ["age_group", "trips"]
            context["summary"] = "Trip distribution by age group."
            context.setdefault("highlights", []).append(
                f"Top age group: {counts.iloc[0, 0]} ({int(counts.iloc[0, 1])} trips)"
            )
            # Bar chart
            import plotly.express as px

            fig = px.bar(counts, x="age_group", y="trips", title="Trips by Age Group")
            figures.append(fig)
        else:
            context["summary"] = "I couldn't match riders to demographics for this filter."
            figures.append(plot_time_series(filtered))

    elif intent == "operational":
        size_col = None
        for c in ["num_riders", "group_size", "riders", "party_size"]:
            if c in trips.columns:
                size_col = c
                break
        if size_col:
            dist = filtered[size_col].dropna().astype(int)
            context["summary"] = "Group size distribution overview."
            if not dist.empty:
                context.setdefault("highlights", []).append(
                    f"Median group size: {int(dist.median())} (p95: {int(np.percentile(dist, 95))})"
                )
                import plotly.express as px

                fig = px.histogram(filtered, x=size_col, nbins=12, title="Group Size Distribution")
                figures.append(fig)
            else:
                figures.append(plot_time_series(filtered))
        else:
            context["summary"] = "I couldn't find a group size column in the data."
            figures.append(plot_time_series(filtered))

    else:  # general
        context["summary"] = "Here's an overview based on recent trips."
        figures.append(plot_time_series(filtered))

    # Numeric summary
    context["stats"].update({
        "Trips (filtered)": int(len(filtered)),
        "Date range": (
            f"{filtered['event_time'].min()} → {filtered['event_time'].max()}"
            if "event_time" in filtered.columns and not filtered.empty
            else "n/a"
        ),
    })

    return context, figures
