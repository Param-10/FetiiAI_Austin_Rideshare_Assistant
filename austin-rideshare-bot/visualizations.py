from __future__ import annotations

from typing import Optional

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_time_series(trips: pd.DataFrame, time_col: str = "event_time") -> go.Figure:
    df = trips.copy()
    if time_col not in df or df.empty:
        return go.Figure()
    # Coerce and drop NaT before resampling
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    if df.empty:
        return go.Figure()
    df = df.sort_values(time_col)
    ts = (
        df.set_index(time_col)
        .assign(count=1)
        .resample("D")["count"]
        .sum()
        .reset_index()
    )
    fig = px.line(ts, x=time_col, y="count", title="Trips Over Time")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig


def plot_top_locations(
    trips: pd.DataFrame,
    address_col: str = "dropoff_address",
    lat_col: str = "dropoff_latitude",
    lon_col: str = "dropoff_longitude",
    top_n: int = 10,
) -> go.Figure:
    df = trips.copy()
    if df.empty:
        return go.Figure()
    if address_col in df.columns and df[address_col].notna().any():
        counts = (
            df[~df[address_col].isna()][address_col]
            .astype(str)
            .str.strip()
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        counts.columns = ["location", "trips"]
    elif {lat_col, lon_col}.issubset(df.columns):
        # Round lat/lon to cluster
        round_lat = df[lat_col].round(3)
        round_lon = df[lon_col].round(3)
        counts = (
            pd.DataFrame({"location": round_lat.astype(str) + ", " + round_lon.astype(str)})
            .value_counts()
            .head(top_n)
            .reset_index(name="trips")
        )
    else:
        return go.Figure()

    fig = px.bar(counts, x="location", y="trips", title=f"Top {top_n} Drop-off Locations")
    fig.update_layout(xaxis_tickangle=-30, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def plot_hourly_pattern(trips: pd.DataFrame) -> go.Figure:
    df = trips.copy()
    if df.empty or "hour" not in df.columns:
        return go.Figure()
    counts = df["hour"].value_counts().sort_index().reset_index()
    counts.columns = ["hour", "trips"]
    fig = px.bar(counts, x="hour", y="trips", title="Hourly Pickup Pattern")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig


def make_map_html(
    trips: pd.DataFrame,
    lat_col: str = "dropoff_latitude",
    lon_col: str = "dropoff_longitude",
    max_points: int = 800,
    title: Optional[str] = None,
) -> str:
    df = trips.copy()
    if df.empty or not {lat_col, lon_col}.issubset(df.columns):
        return ""
    df = df.dropna(subset=[lat_col, lon_col]).head(max_points)
    if df.empty:
        return ""

    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=(row[lat_col], row[lon_col]),
            radius=3,
            color="#2563eb",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.6,
            opacity=0.6,
        ).add_to(m)

    if title:
        folium.map.CustomPane("title").add_to(m)

    return m._repr_html_()
