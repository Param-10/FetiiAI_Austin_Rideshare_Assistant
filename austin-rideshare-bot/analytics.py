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


def _get_current_theme() -> str:
    """Get current theme from streamlit session state."""
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and 'theme' in st.session_state:
            return st.session_state['theme']
    except Exception:
        pass
    return "dark"  # Default theme


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
                # Handle timezone-aware datetimes properly to avoid warnings
                if max_time.tz:
                    # Remove timezone info before converting to period
                    max_time_naive = max_time.tz_localize(None)
                    target_period = (max_time_naive - pd.offsets.MonthBegin(1)).to_period("M")
                    # Remove timezone from event_time before converting to period
                    event_time_naive = df["event_time"].dt.tz_localize(None)
                    df = df[event_time_naive.dt.to_period("M") == target_period]
                else:
                    target_period = (max_time - pd.offsets.MonthBegin(1)).to_period("M")
                    df = df[df["event_time"].dt.to_period("M") == target_period]
            elif entities["relative_time"] == "this_month":
                # Handle timezone-aware datetimes properly to avoid warnings
                if max_time.tz:
                    # Remove timezone info before converting to period
                    max_time_naive = max_time.tz_localize(None)
                    target_period = max_time_naive.to_period("M")
                    # Remove timezone from event_time before converting to period
                    event_time_naive = df["event_time"].dt.tz_localize(None)
                    df = df[event_time_naive.dt.to_period("M") == target_period]
                else:
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


def _analyze_short_routes(df: pd.DataFrame, max_distance: float = 3.0) -> List[str]:
    """Analyze common short routes (under specified km) for scooter-friendly insights."""
    if df.empty or 'trip_distance_km' not in df.columns:
        return []
    
    short_trips = df[df['trip_distance_km'] <= max_distance]
    if short_trips.empty:
        return []
    
    routes = []
    
    # Check if we have both pickup and dropoff addresses
    if ('pickup_address' in short_trips.columns and 'dropoff_address' in short_trips.columns):
        route_pairs = short_trips.dropna(subset=['pickup_address', 'dropoff_address'])
        if not route_pairs.empty:
            # Create route strings and count frequencies
            route_pairs['route'] = route_pairs['pickup_address'].str[:30] + " → " + route_pairs['dropoff_address'].str[:30]
            top_routes = route_pairs['route'].value_counts().head(3)
            
            for route, count in top_routes.items():
                if count >= 2:  # Only show routes with multiple occurrences
                    distance_avg = route_pairs[route_pairs['route'] == route]['trip_distance_km'].mean()
                    routes.append(f"{route} ({count} trips, avg {distance_avg:.1f}km)")
    
    # Fallback to just dropoff locations if no route pairs
    if not routes:
        top_short_drops = _top_locations(short_trips, top_n=3)
        for location, count in top_short_drops:
            if count >= 2:
                avg_dist = short_trips[short_trips['dropoff_address'].str.contains(location[:20], na=False)]['trip_distance_km'].mean()
                routes.append(f"{location[:40]} ({count} short trips, avg {avg_dist:.1f}km)")
    
    return routes[:3]  # Return top 3 short routes


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
    conversation_context: Dict[str, Any] = None,
) -> Tuple[Dict[str, Any], List[Any]]:
    """Return (context, figures)."""
    if trips is None or trips.empty:
        return ({"summary": "I couldn't find trip data.", "highlights": []}, [])

    # Chat-only: do not return figures
    figures: List[Any] = []
    context: Dict[str, Any] = {"stats": {}}
    conversation_context = conversation_context or {}

    # Handle continuing previous question
    if intent == "continue_previous":
        last_topic = conversation_context.get("last_question_topic", "")
        if last_topic == "group_sizes":
            # Override intent to focus on group sizes/operational data
            intent = "operational"
            # Add context that this is a continuation
            context["summary"] = "Perfect! Let me show you the group size patterns."
        elif last_topic == "timing":
            intent = "temporal"
            context["summary"] = "Great! Here are the timing patterns."
        elif last_topic == "locations":
            intent = "location"
            context["summary"] = "Awesome! Let me show you the location data."
        elif last_topic == "demographics":
            intent = "demographic"
            context["summary"] = "Sure! Here's the demographic breakdown."
        else:
            # Fallback if no previous topic tracked
            intent = "casual_response"

    # Get current theme for consistent styling
    current_theme = _get_current_theme()
    
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

        # Top locations list with more detail
        tops = _top_locations(loc_df)
        if tops:
            location_details = []
            for name, cnt in tops[:3]:  # Focus on top 3 for specificity
                if len(name) > 50:  # If address is long, extract key part
                    import re
                    # Try to extract street name or landmark
                    street_match = re.search(r'(\d+\s+\w+\s+(St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road))', name)
                    if street_match:
                        name = street_match.group(1)
                    else:
                        name = name[:30] + "..."
                location_details.append(f"{name} ({cnt} trips)")
            
            context.setdefault("highlights", []).append(
                "Top destinations: " + ", ".join(location_details)
            )
            
            # Add insight about short vs long trips if relevant
            if 'trip_distance_km' in loc_df.columns:
                short_trips = loc_df[loc_df['trip_distance_km'] < 3].shape[0]
                total_trips = loc_df.shape[0]
                if short_trips > 0:
                    context.setdefault("highlights", []).append(
                        f"{short_trips} of {total_trips} trips were under 3km (perfect for e-scooters or walking)"
                    )
                    
                    # Add specific short route analysis
                    short_routes = _analyze_short_routes(loc_df)
                    if short_routes:
                        context.setdefault("highlights", []).append(
                            "Popular short routes: " + "; ".join(short_routes[:2])
                        )

        # Chat-only mode: no figures

    elif intent == "temporal":
        tdf = filtered
        if tdf.empty:
            # fallback: show overall if filters eliminate all data
            tdf = trips.copy()
            context["summary"] = "No trips matched the time filters. Showing overall timing patterns."
        else:
            context["summary"] = "Here's when people are riding."

        # Focus on TIMING information for temporal queries
        if not tdf.empty:
            # Peak hours - most important for "when" questions
            peaks = _peak_hours(tdf, top_k=3)
            if peaks:
                if len(peaks) >= 2:
                    context.setdefault("highlights", []).append(
                        f"Peak hours: {peaks[0]}:00, {peaks[1]}:00, and {peaks[2]}:00" if len(peaks) >= 3 else f"Peak hours: {peaks[0]}:00 and {peaks[1]}:00"
                    )
                else:
                    context.setdefault("highlights", []).append(f"Peak hour: {peaks[0]}:00")
            
            # Peak days if available
            if "day_of_week" in tdf.columns:
                vc = tdf["day_of_week"].dropna().value_counts()
                if not vc.empty and len(vc) > 1:
                    top_days = vc.head(2)
                    if len(top_days) >= 2:
                        context.setdefault("highlights", []).append(
                            f"Busiest days: {top_days.index[0]} ({top_days.iloc[0]} trips) and {top_days.index[1]} ({top_days.iloc[1]} trips)"
                        )
                    else:
                        context.setdefault("highlights", []).append(f"Peak day: {top_days.index[0]} ({top_days.iloc[0]} trips)")
            
            # Weekend vs weekday pattern if available
            if "is_weekend" in tdf.columns:
                weekend_count = len(tdf[tdf["is_weekend"] == True])
                weekday_count = len(tdf[tdf["is_weekend"] == False])
                if weekend_count > 0 and weekday_count > 0:
                    if weekend_count > weekday_count:
                        context.setdefault("highlights", []).append(f"More weekend activity: {weekend_count} weekend trips vs {weekday_count} weekday trips")
                    else:
                        context.setdefault("highlights", []).append(f"More weekday activity: {weekday_count} weekday trips vs {weekend_count} weekend trips")

        # Chat-only: no figures

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
                # Chat-only: no figures
            else:
                context["summary"] = "I couldn't find trips matching that age range."
                figures.append(plot_time_series(filtered, theme=current_theme))
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
                # Chat-only: no figures
            else:
                context["summary"] = "I couldn't match riders to demographics for this filter."
                figures.append(plot_time_series(filtered, theme=current_theme))

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
                # Chat-only: no figures
            else:
                pass
        else:
            context["summary"] = "I couldn't find a group size column in the data."
            pass

    elif intent == "casual_greeting":
        # Handle friendly greetings - NO DATA ANALYSIS
        context["summary"] = "What's up! Ready to explore Austin's ride patterns?"
        
        import random
        greeting_responses = [
            "I've got fresh insights from over 2,000 group rides around Austin!",
            "Let's dive into some Austin transportation data together!",
            "I'm here with the latest scoop on how people get around our city!"
        ]
        context.setdefault("highlights", []).append(random.choice(greeting_responses))
        
        # Skip all data processing for casual greetings - return early
        context["stats"] = {}  # No stats for greetings
        return context, figures
    
    elif intent == "casual_response":
        # Handle casual "ok", "cool", etc. - only give interesting info if there's something fresh
        context["summary"] = "Cool! What would you like to explore about Austin's ride patterns?"
        
        import random
        used_stats = conversation_context.get("mentioned_stats", set()) if conversation_context else set()
        
        # Only give ONE fresh insight if available, otherwise just ask what they want
        if not filtered.empty and len(used_stats) < 3:  # Only if we haven't shared much yet
            tops = _top_locations(filtered, top_n=3)
            if tops:
                top_spot = tops[0][0]
                clean_name = top_spot.split(',')[0] if ',' in top_spot else top_spot[:40] 
                insight = f"{clean_name} is popular with {tops[0][1]} recent trips"
                if insight not in used_stats:
                    context.setdefault("highlights", []).append(insight)
                else:
                    context.setdefault("highlights", []).append("What specific area or time period interests you?")
            else:
                context.setdefault("highlights", []).append("What aspect of Austin transportation interests you most?")
        else:
            # Ask what they want instead of dumping data
            context.setdefault("highlights", []).append("What would you like to know about Austin rides?")
        
        # Minimal stats for casual responses
        context["stats"] = {"Available": "Data ready for your questions"}
        return context, figures
    
    elif intent == "confused":
        # Handle confusion - clarify and offer help - NO DATA ANALYSIS
        context["summary"] = "Sorry for the confusion! Let me help you explore Austin ride data more clearly."
        
        context.setdefault("highlights", []).append(
            "I can help you discover things like: popular destinations, peak hours, group sizes, costs, or specific neighborhoods. What sounds interesting?"
        )
        
        # No stats for confused responses - just help clarification  
        context["stats"] = {}
        return context, figures
    
    elif intent == "conversational":
        # Handle other casual responses - keep it flowing naturally
        context["summary"] = "Just vibes, checking out the Austin ride scene."
        
        # Give varied interesting facts to keep conversation going
        import random
        
        interesting_facts = []
        if not filtered.empty:
            tops = _top_locations(filtered, top_n=5)
            if tops:
                # Random interesting location fact
                random_spot = random.choice(tops[:3])
                interesting_facts.append(f"Mad people hit up {random_spot[0][:30]}... - {random_spot[1]} trips there")
            
            if 'trip_distance_km' in filtered.columns:
                short_routes = _analyze_short_routes(filtered)
                if short_routes:
                    interesting_facts.append(f"Popular short hop: {short_routes[0]}")
                    
                # Add distance variety fact
                distances = filtered['trip_distance_km'].dropna()
                if not distances.empty:
                    under_2km = len(distances[distances < 2])
                    over_10km = len(distances[distances > 10])
                    if under_2km > 0:
                        interesting_facts.append(f"{under_2km} quick trips under 2km (perfect scooter weather)")
                    if over_10km > 0:
                        interesting_facts.append(f"{over_10km} long hauls over 10km")
        
        if interesting_facts:
            context.setdefault("highlights", interesting_facts[:2])
    
    elif intent == "more_info":
        context["summary"] = "Let me dive deeper into the Austin transportation scene."
        
        # Provide additional insights they haven't heard yet
        if not filtered.empty:
            # Peak times insight
            if 'hour' in filtered.columns:
                peak_hours = _peak_hours(filtered, top_k=2)
                if peak_hours:
                    context.setdefault("highlights", []).append(
                        f"Peak riding hours: {peak_hours[0]}:00 and {peak_hours[1]}:00" if len(peak_hours) >= 2 else f"Peak hour: {peak_hours[0]}:00"
                    )
            
            # Weekend vs weekday pattern
            if 'is_weekend' in filtered.columns:
                weekend_trips = len(filtered[filtered['is_weekend'] == True])
                weekday_trips = len(filtered[filtered['is_weekend'] == False]) 
                if weekend_trips > 0 and weekday_trips > 0:
                    context.setdefault("highlights", []).append(
                        f"Weekend trips: {weekend_trips} vs Weekday: {weekday_trips}"
                    )
            
            # Cost insights
            if 'estimated_cost_usd' in filtered.columns:
                costs = filtered['estimated_cost_usd'].dropna()
                if not costs.empty:
                    cheap_trips = len(costs[costs < 10])
                    expensive_trips = len(costs[costs > 20])
                    if cheap_trips > 0:
                        context.setdefault("highlights", []).append(
                            f"{cheap_trips} trips under $10 (budget friendly), {expensive_trips} over $20"
                        )

    else:  # general
        import random
        
        # Vary the summary message
        summary_options = [
            "Here's what I found in the Austin ride data.",
            "Looking at recent trip patterns around town.",
            "Here's an overview of the ride activity.",
            "Digging into the Austin transportation scene."
        ]
        context["summary"] = random.choice(summary_options)
        
        # For general queries, provide diverse insights with better variety
        if not filtered.empty:
            # Randomly choose what type of insight to lead with
            insight_types = []
            
            # Top destinations - but vary the format
            tops = _top_locations(filtered, top_n=4)
            if tops:
                if len(tops) >= 3:
                    # Sometimes show just the top spot, sometimes show multiple
                    if random.choice([True, False]):
                        top_spot = tops[0][0]
                        # Extract a cleaner name
                        clean_name = top_spot.split(',')[0] if ',' in top_spot else top_spot[:40]
                        insight_types.append(f"{clean_name} is the hottest spot right now ({tops[0][1]} trips)")
                    else:
                        top_three = ", ".join([
                            (name.split(',')[0] if ',' in name else name[:25]) 
                            for name, cnt in tops[:3]
                        ])
                        insight_types.append(f"Top spots: {top_three}")
            
            # Distance/route insights
            if 'trip_distance_km' in filtered.columns:
                distances = filtered['trip_distance_km'].dropna()
                if not distances.empty:
                    avg_dist = distances.mean()
                    short_count = len(distances[distances < 2])
                    long_count = len(distances[distances > 10])
                    
                    if short_count > long_count and short_count > 10:
                        insight_types.append(f"Lots of short hops - {short_count} trips under 2km")
                    elif avg_dist > 8:
                        insight_types.append(f"People are traveling far - average distance is {avg_dist:.1f}km")
                    else:
                        insight_types.append(f"Mix of distances - average trip is {avg_dist:.1f}km")
                        
                    # Add specific short routes sometimes
                    short_routes = _analyze_short_routes(filtered)
                    if short_routes and random.choice([True, False]):
                        insight_types.append(f"Popular quick trip: {short_routes[0]}")
            
            # Time-based insights
            if 'hour' in filtered.columns:
                peak_hours = _peak_hours(filtered, top_k=2)
                if peak_hours:
                    if len(peak_hours) >= 2:
                        insight_types.append(f"Busiest times: {peak_hours[0]}:00 and {peak_hours[1]}:00")
                    else:
                        insight_types.append(f"Peak hour: {peak_hours[0]}:00")
            
            # Cost insights if available
            if 'estimated_cost_usd' in filtered.columns:
                costs = filtered['estimated_cost_usd'].dropna()
                if not costs.empty:
                    avg_cost = costs.mean()
                    if avg_cost < 12:
                        insight_types.append(f"Budget friendly rides - averaging ${avg_cost:.2f}")
                    elif avg_cost > 20:
                        insight_types.append(f"Higher cost trips - averaging ${avg_cost:.2f}")
                    else:
                        insight_types.append(f"Average ride costs ${avg_cost:.2f}")
            
            # Group size insights
            if 'num_riders' in filtered.columns:
                group_sizes = filtered['num_riders'].dropna()
                if not group_sizes.empty:
                    big_groups = len(group_sizes[group_sizes >= 6])
                    if big_groups > 5:
                        insight_types.append(f"{big_groups} big group adventures (6+ people)")
                    
            # Pick 1-2 random insights to avoid always showing the same pattern
            if insight_types:
                selected_insights = random.sample(insight_types, min(2, len(insight_types)))
                for insight in selected_insights:
                    context.setdefault("highlights", []).append(insight)

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
