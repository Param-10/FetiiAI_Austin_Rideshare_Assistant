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


def _calculate_trip_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Add trip distance calculations to dataframe"""
    df = df.copy()
    
    # Use the correct column names based on actual data
    pickup_lat_col = 'pickup_latitude' if 'pickup_latitude' in df.columns else 'pick_up_latitude'
    pickup_lon_col = 'pickup_longitude' if 'pickup_longitude' in df.columns else 'pick_up_longitude'
    dropoff_lat_col = 'dropoff_latitude' if 'dropoff_latitude' in df.columns else 'drop_off_latitude'
    dropoff_lon_col = 'dropoff_longitude' if 'dropoff_longitude' in df.columns else 'drop_off_longitude'
    
    required_cols = [pickup_lat_col, pickup_lon_col, dropoff_lat_col, dropoff_lon_col]
    
    if all(col in df.columns for col in required_cols):
        # Calculate distances for each trip
        distances = []
        for idx, row in df.iterrows():
            if (pd.notna(row.get(pickup_lat_col)) and pd.notna(row.get(pickup_lon_col)) and
                pd.notna(row.get(dropoff_lat_col)) and pd.notna(row.get(dropoff_lon_col))):
                dist = _haversine_km(
                    row[pickup_lat_col], row[pickup_lon_col],
                    row[dropoff_lat_col], row[dropoff_lon_col]
                )
                distances.append(dist)
            else:
                distances.append(np.nan)
        
        df['trip_distance_km'] = distances
    else:
        df['trip_distance_km'] = np.nan
    
    return df


def _estimate_trip_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Add estimated trip cost calculations"""
    df = df.copy()
    
    # Austin rideshare pricing model (approximate)
    BASE_FARE = 2.50
    COST_PER_KM = 1.20
    COST_PER_MINUTE = 0.25
    SURGE_MULTIPLIER = 1.0  # Could be dynamic based on time/location
    
    # Estimate trip duration based on distance (avg 30 km/h in city)
    df['estimated_duration_min'] = df.get('trip_distance_km', 0) * 2.0  # 30 km/h = 2 min/km
    
    # Calculate estimated cost
    distance_cost = df.get('trip_distance_km', 0) * COST_PER_KM
    time_cost = df.get('estimated_duration_min', 0) * COST_PER_MINUTE
    
    df['estimated_cost_usd'] = (BASE_FARE + distance_cost + time_cost) * SURGE_MULTIPLIER
    df['cost_per_passenger'] = df['estimated_cost_usd'] / df.get('num_riders', 1)
    
    return df


def _analyze_route_efficiency(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze route efficiency and optimization opportunities"""
    if 'trip_distance_km' not in df.columns:
        return {}
    
    valid_distances = df['trip_distance_km'].dropna()
    if len(valid_distances) == 0:
        return {}
    
    analysis = {
        'avg_trip_distance': f"{valid_distances.mean():.2f} km",
        'median_trip_distance': f"{valid_distances.median():.2f} km",
        'longest_trip': f"{valid_distances.max():.2f} km",
        'shortest_trip': f"{valid_distances.min():.2f} km",
    }
    
    # Identify potentially inefficient routes (very short or very long)
    short_trips = len(valid_distances[valid_distances < 1.0])  # < 1km
    long_trips = len(valid_distances[valid_distances > 20.0])  # > 20km
    
    if short_trips > 0:
        analysis['short_trips'] = f"{short_trips} trips under 1km (consider walking/biking)"
    if long_trips > 0:
        analysis['long_trips'] = f"{long_trips} trips over 20km (consider alternative transport)"
    
    return analysis


def _within_place(df: pd.DataFrame, place: Place, lat_col: str, lon_col: str) -> pd.Series:
    if df.empty or not {lat_col, lon_col}.issubset(df.columns):
        return pd.Series([False] * len(df), index=df.index)
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    mask = lat.notna() & lon.notna()
    if not mask.any():
        return pd.Series([False] * len(df), index=df.index)

    lat_rad = np.radians(lat[mask].to_numpy())
    lon_rad = np.radians(lon[mask].to_numpy())
    plat = math.radians(place.lat)
    plon = math.radians(place.lon)

    dphi = lat_rad - plat
    dlambda = lon_rad - plon
    a = np.sin(dphi / 2) ** 2 + np.cos(lat_rad) * np.cos(plat) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist_km = 6371.0 * c

    out = pd.Series([False] * len(df), index=df.index)
    out.loc[mask] = dist_km <= place.radius_km
    return out


def _apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters from session state"""
    try:
        import streamlit as st
        
        # Apply date filter
        if hasattr(st, 'session_state') and 'date_filter' in st.session_state:
            date_range = st.session_state['date_filter']
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start_date, end_date = date_range
                if 'event_time' in df.columns:
                    df = df[
                        (df['event_time'].dt.date >= start_date) & 
                        (df['event_time'].dt.date <= end_date)
                    ]
        
        # Apply rider count filter
        if hasattr(st, 'session_state') and 'rider_filter' in st.session_state:
            rider_range = st.session_state['rider_filter']
            if isinstance(rider_range, (list, tuple)) and len(rider_range) == 2:
                min_riders, max_riders = rider_range
                if 'num_riders' in df.columns:
                    df = df[
                        (df['num_riders'] >= min_riders) & 
                        (df['num_riders'] <= max_riders)
                    ]
        
        return df
    except Exception:
        # If streamlit not available or error, return original df
        return df


def _filter_by_entities(trips: pd.DataFrame, entities: Dict[str, Any]) -> pd.DataFrame:
    df = trips.copy()
    if df.empty:
        return df

    # Time filters
    if "days" in entities:
        days = entities["days"]
        if "day_of_week" in df.columns and df["day_of_week"].notna().any():
            df = df[df["day_of_week"].isin(days)].copy()
        elif "is_weekend" in df.columns:
            weekend_days = {"Friday", "Saturday", "Sunday"}
            if set(d.capitalize() for d in days) & weekend_days:
                df = df[df["is_weekend"] == True].copy()

    if "hour_range" in entities and "hour" in df.columns:
        start, end = entities["hour_range"]
        if start <= end:
            df = df[(df["hour"] >= start) & (df["hour"] <= end)]
        else:
            # Handle wrap-around (20→03)
            df = df[(df["hour"] >= start) | (df["hour"] <= end)]

    # Relative time filters (skip if no valid timestamps)
    if "relative_time" in entities and "event_time" in df.columns:
        t = df["event_time"].dropna()
        if pd.api.types.is_datetime64_any_dtype(df["event_time"]) and not t.empty:
            max_time = t.max()
            if entities["relative_time"] == "last_month":
                target_period = (max_time - pd.offsets.MonthBegin(1)).to_period("M")
                df = df[df["event_time"].dt.to_period("M") == target_period]
            elif entities["relative_time"] == "this_month":
                target_period = max_time.to_period("M")
                df = df[df["event_time"].dt.to_period("M") == target_period]
            elif entities["relative_time"] == "last_week":
                week_start = (max_time.normalize() - pd.Timedelta(days=7))
                df = df[(df["event_time"] >= week_start) & (df["event_time"] <= max_time)]
        # else: no-op if event_time all NaT

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

    # Join riders per trip (trip_id, user_id) -> demographics (user_id -> age)
    if not riders.empty and not demo.empty and {"trip_id", "user_id"}.issubset(riders.columns) and {"user_id"}.issubset(demo.columns):
        riders_join = riders.merge(demo[["user_id", "age", "age_group"]], on="user_id", how="left")
        df = df.merge(riders_join, on="trip_id", how="left")
        return df

    # Fallback: use booker user id if available
    if "booker_user_id" in df.columns and not demo.empty and "user_id" in demo.columns:
        return df.merge(demo[["user_id", "age", "age_group"]], left_on="booker_user_id", right_on="user_id", how="left")

    return df


def _apply_age_filter(df: pd.DataFrame, riders: pd.DataFrame, demo: pd.DataFrame, entities: Dict[str, Any]) -> pd.DataFrame:
    """Filter trips to those containing at least one rider within age range specified in entities."""
    if ("age_min" not in entities and "age_max" not in entities) or df.empty:
        return df
    jdf = _join_with_demographics(df, riders, demo)
    if jdf.empty or "age" not in jdf.columns:
        return df.iloc[0:0]  # no matching info
    amin = entities.get("age_min", -np.inf)
    amax = entities.get("age_max", np.inf)
    # Keep trips where any rider in range
    mask = (jdf["age"].astype(float) >= amin) & (jdf["age"].astype(float) <= amax)
    keep_trip_ids = jdf.loc[mask, "trip_id"].dropna().unique().tolist() if "trip_id" in jdf.columns else []
    if not keep_trip_ids:
        return df.iloc[0:0]
    return df[df["trip_id"].isin(keep_trip_ids)].copy()


def _top_locations(df: pd.DataFrame, address_col: str = "dropoff_address", loc_col: str = "dropoff_loc", top_n: int = 5) -> List[Tuple[str, int]]:
    if df.empty:
        return []
    if address_col in df.columns and df[address_col].notna().any():
        counts = (
            df[~df[address_col].isna()][address_col].astype(str).str.strip().value_counts().head(top_n)
        )
        return [(idx, int(val)) for idx, val in counts.items()]
    if loc_col in df.columns and df[loc_col].notna().any():
        counts = df[loc_col].astype(str).value_counts().head(top_n)
        return [(idx, int(val)) for idx, val in counts.items()]
    return []


def _peak_hours(df: pd.DataFrame, top_k: int = 3) -> List[int]:
    if df.empty or "hour" not in df.columns:
        return []
    vc = df["hour"].value_counts().sort_values(ascending=False)
    return [int(h) for h in vc.head(top_k).index.tolist()]


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

    # Add advanced analytics to trips data
    trips_with_analytics = _calculate_trip_distances(trips)
    trips_with_analytics = _estimate_trip_costs(trips_with_analytics)

    # Apply sidebar filters first
    trips_filtered = _apply_sidebar_filters(trips_with_analytics)

    # Pre-filter by entities
    filtered = _filter_by_entities(trips_filtered, entities)

    if intent == "location":
        # Optional place filter
        place_key = entities.get("place")
        loc_df = filtered
        if place_key and place_key in KNOWN_PLACES:
            place = KNOWN_PLACES[place_key]
            mask = _within_place(loc_df, place, "dropoff_latitude", "dropoff_longitude")
            loc_df = loc_df[mask].copy()
            title = f"Trips near {place.name}"
        else:
            title = "Popular Drop-offs"

        # Optional age filter
        loc_df = _apply_age_filter(loc_df, riders, demographics, entities)

        summary = _summarize_count(loc_df, label=title)
        context["summary"] = summary
        context.setdefault("highlights", []).append(summary)

        # Top locations list
        tops = _top_locations(loc_df)
        if tops:
            context.setdefault("highlights", []).append(
                "Top locations: " + ", ".join([f"{name} ({cnt})" for name, cnt in tops])
            )

        # Bar chart
        bar = plot_top_locations(loc_df)
        figures.append(bar)

        # Map
        mhtml = make_map_html(loc_df)
        if mhtml:
            figures.append(mhtml)

    elif intent == "temporal":
        tdf = filtered
        if tdf.empty:
            # fallback: show overall if filters eliminate all data
            tdf = trips.copy()
            context["summary"] = "No trips matched the time filters. Showing overall pattern."
        else:
            summary = _summarize_count(tdf, label="Filtered trips")
            context["summary"] = "Here's how demand changes over time."
            context.setdefault("highlights", []).append(summary)

        # Add peak day/hour if available
        if not tdf.empty:
            if "day_of_week" in tdf.columns:
                vc = tdf["day_of_week"].dropna().value_counts()
                if not vc.empty:
                    top_day = vc.idxmax()
                    context.setdefault("highlights", []).append(f"Peak day: {top_day}")
            peaks = _peak_hours(tdf)
            if peaks:
                context.setdefault("highlights", []).append(
                    "Peak hours: " + ", ".join(str(h) for h in peaks)
                )

        ts_fig = plot_time_series(tdf)
        hr_fig = plot_hourly_pattern(tdf)
        figures.extend([ts_fig, hr_fig])

    elif intent == "demographic":
        # If a specific age range is requested, filter and show top locations for that group
        if ("age_min" in entities or "age_max" in entities):
            jdf = _apply_age_filter(filtered, riders, demographics, entities)
            if not jdf.empty:
                context["summary"] = "Top drop-offs for the requested age range."
                tops = _top_locations(jdf)
                if tops:
                    context.setdefault("highlights", []).append(
                        "Top locations: " + ", ".join([f"{name} ({cnt})" for name, cnt in tops])
                    )
                figures.append(plot_top_locations(jdf))
                mhtml = make_map_html(jdf)
                if mhtml:
                    figures.append(mhtml)
            else:
                context["summary"] = "I couldn't find trips matching that age range."
                figures.append(plot_time_series(filtered))
        else:
            # Otherwise show distribution by age group
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
                # If a place is specified, also highlight peak hours for large groups
                place_key = entities.get("place")
                if place_key and place_key in KNOWN_PLACES:
                    place = KNOWN_PLACES[place_key]
                    mask = _within_place(filtered, place, "dropoff_latitude", "dropoff_longitude")
                    place_df = filtered[mask].copy()
                    peaks = _peak_hours(place_df)
                    if peaks:
                        context.setdefault("highlights", []).append(
                            f"Peak hours near {place.name}: " + ", ".join(str(h) for h in peaks)
                        )
                import plotly.express as px
                fig = px.histogram(filtered, x=size_col, nbins=12, title="Group Size Distribution")
                figures.append(fig)
                figures.append(plot_hourly_pattern(filtered))
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

    # Add advanced analytics to context
    if not filtered.empty:
        # Route efficiency analysis
        route_analysis = _analyze_route_efficiency(filtered)
        if route_analysis:
            context["stats"].update(route_analysis)
        
        # Cost analysis
        if 'estimated_cost_usd' in filtered.columns:
            valid_costs = filtered['estimated_cost_usd'].dropna()
            if len(valid_costs) > 0:
                context["stats"]["avg_trip_cost"] = f"${valid_costs.mean():.2f}"
                context["stats"]["avg_cost_per_passenger"] = f"${filtered['cost_per_passenger'].dropna().mean():.2f}"
                
        # Distance analysis
        if 'trip_distance_km' in filtered.columns:
            valid_distances = filtered['trip_distance_km'].dropna()
            if len(valid_distances) > 0:
                total_distance = valid_distances.sum()
                context["stats"]["total_distance"] = f"{total_distance:.1f} km"

    return context, figures
