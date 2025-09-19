from __future__ import annotations

from typing import Optional, Dict, Any

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# --- Helper Functions ---
def _get_theme_colors(theme: str = "dark") -> Dict[str, str]:
    """Get color scheme based on theme."""
    if theme == "light":
        return {
            'bg': '#FFFFFF',
            'text': '#0E1117',
            'grid': '#E5E7EB',
            'border': '#D1D5DB',
            'accent': '#1D4ED8'
        }
    else:  # dark theme
        return {
            'bg': '#0E1117',
            'text': '#FAFAFA', 
            'grid': '#374151',
            'border': '#4B5563',
            'accent': '#3B82F6'
        }


def _create_empty_figure(title: str, message: str, theme: str = "dark") -> go.Figure:
    """Create a styled empty figure with an informative message."""
    colors = _get_theme_colors(theme)
    
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f"<b>{message}</b><br><br>Try adjusting your filters or query",
        showarrow=False,
        xref="paper", yref="paper",
        font=dict(size=16, color=colors['text']),
        bgcolor=colors['bg'],
        bordercolor=colors['border'],
        borderwidth=1
    )
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': colors['text']}
        },
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor=colors['bg'],
        paper_bgcolor=colors['bg'],
        height=400,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig


def _normalize_lat_lon(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """Return a copy ensuring numeric lat/lon and auto-fix swapped columns when detected.

    This function coerces latitude and longitude to numeric values and then
    detects rows where values look swapped (e.g., latitude around -97 and
    longitude around 30 in Austin). It swaps those values back.
    """
    out = df.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")

    # Detect obvious invalid ranges
    invalid_range = (out[lat_col].abs() > 90) | (out[lon_col].abs() > 180)

    # Detect common Austin swap: lat near -97, lon near 30
    likely_swapped = (
        out[lat_col].between(-98.5, -96.5, inclusive="both") &
        out[lon_col].between(29.0, 31.5, inclusive="both")
    )

    swap_mask = invalid_range | likely_swapped
    if swap_mask.any():
        out.loc[swap_mask, [lat_col, lon_col]] = out.loc[swap_mask, [lon_col, lat_col]].values
        # Re-coerce after swap
        out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
        out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")

    return out


def plot_time_series(trips: pd.DataFrame, time_col: str = "event_time", theme: str = "dark") -> go.Figure:
    """Create an improved time series plot with better styling and error handling."""
    df = trips.copy()
    
    # Better error handling with informative messages
    if df.empty:
        return _create_empty_figure("Trips Over Time", "No trip data available", theme)
        
    if time_col not in df.columns:
        return _create_empty_figure("Trips Over Time", f"Column '{time_col}' not found in data", theme)
        
    if df[time_col].isna().all():
        return _create_empty_figure("Trips Over Time", "All timestamps are missing", theme)

    # Coerce and drop NaT before resampling
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    if df.empty:
        return _create_empty_figure("Trips Over Time", "No valid timestamps found", theme)

    # Create a date column for grouping
    df['date'] = df[time_col].dt.date

    # Group by date and count trips
    ts = df.groupby('date').size().reset_index(name='trips')
    ts = ts.rename(columns={'date': 'Date', 'trips': 'Number of Trips'})
    
    # Ensure the Date column is sorted for the line chart
    ts = ts.sort_values(by='Date')
    
    # Calculate some statistics for better insights
    avg_trips = ts['Number of Trips'].mean()
    max_trips = ts['Number of Trips'].max()
    max_date = ts.loc[ts['Number of Trips'].idxmax(), 'Date']

    # Create the figure with better styling
    fig = go.Figure()
    
    # Add main line
    fig.add_trace(go.Scatter(
        x=ts['Date'],
        y=ts['Number of Trips'],
        mode='lines+markers',
        name='Daily Trips',
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=6, color='#1E40AF'),
        hovertemplate='<b>%{x}</b><br>Trips: %{y}<extra></extra>'
    ))
    
    # Add average line
    fig.add_hline(
        y=avg_trips, 
        line_dash="dash", 
        line_color="rgba(255, 193, 7, 0.7)",
        annotation_text=f"Average: {avg_trips:.1f} trips/day",
        annotation_position="top right"
    )

    # Enhanced layout with better styling
    colors = _get_theme_colors(theme)
    fig.update_layout(
        title={
            'text': "📈 Trips Over Time",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': colors['text']}
        },
        xaxis=dict(
            title=dict(text="Date", font=dict(color=colors['text'])),
            tickformat="%b %d, %Y",
            gridcolor=colors['grid'],
            tickcolor=colors['text'],
            tickfont=dict(color=colors['text'])
        ),
        yaxis=dict(
            title=dict(text="Number of Trips", font=dict(color=colors['text'])),
            rangemode='tozero',
            gridcolor=colors['grid'],
            tickcolor=colors['text'],
            tickfont=dict(color=colors['text'])
        ),
        plot_bgcolor=colors['bg'],
        paper_bgcolor=colors['bg'],
        font={'color': colors['text']},
        margin=dict(l=60, r=60, t=80, b=60),
        hovermode='x unified',
        height=450
    )
    
    # Add annotation for peak day
    fig.add_annotation(
        x=max_date,
        y=max_trips,
        text=f"Peak: {max_trips} trips",
        showarrow=True,
        arrowhead=2,
        arrowcolor=colors['accent'],
        bgcolor=colors['accent'],
        bordercolor=colors['accent'],
        font={'color': 'white'}
    )
    
    return fig


def plot_top_locations(
    trips: pd.DataFrame,
    address_col: str = "dropoff_address",
    lat_col: str = "dropoff_latitude",
    lon_col: str = "dropoff_longitude",
    top_n: int = 10,
    theme: str = "dark"
) -> go.Figure:
    """Create an improved top locations bar chart with better styling and error handling."""
    df = trips.copy()
    
    if df.empty:
        return _create_empty_figure("Top Drop-off Locations", "No trip data available", theme)
    
    counts = None
    
    # Try address column first
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
    # Fallback to coordinates
    elif {lat_col, lon_col}.issubset(df.columns):
        valid_coords = df.dropna(subset=[lat_col, lon_col])
        if not valid_coords.empty:
            # Round lat/lon to cluster
            round_lat = valid_coords[lat_col].round(3)
            round_lon = valid_coords[lon_col].round(3)
            coords_df = pd.DataFrame({"location": round_lat.astype(str) + ", " + round_lon.astype(str)})
            counts = (
                coords_df["location"]
                .value_counts()
                .head(top_n)
                .reset_index()
            )
            counts.columns = ["location", "trips"]
    
    if counts is None or counts.empty:
        return _create_empty_figure("Top Drop-off Locations", "No location data found", theme)

    # Truncate long location names for better display
    counts['location_display'] = counts['location'].apply(
        lambda x: x[:40] + '...' if len(str(x)) > 43 else str(x)
    )
    
    # Create color scale based on trip counts
    colors = _get_theme_colors(theme)
    color_scale = px.colors.sample_colorscale(
        "viridis", [n/(len(counts)-1) for n in range(len(counts))]
    ) if len(counts) > 1 else ['#3B82F6']

    fig = go.Figure(data=[
        go.Bar(
            x=counts['trips'],
            y=counts['location_display'],
            orientation='h',
            marker=dict(
                color=color_scale,
                line=dict(color=colors['border'], width=1)
            ),
            text=counts['trips'],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Trips: %{x}<br><extra></extra>'
        )
    ])

    fig.update_layout(
        title={
            'text': f"🏆 Top {top_n} Drop-off Locations",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': colors['text']}
        },
        xaxis=dict(
            title=dict(text="Number of Trips", font=dict(color=colors['text'])),
            gridcolor=colors['grid'],
            tickcolor=colors['text'],
            tickfont=dict(color=colors['text'])
        ),
        yaxis=dict(
            title=dict(text="Location", font=dict(color=colors['text'])),
            tickcolor=colors['text'],
            tickfont=dict(color=colors['text'])
        ),
        plot_bgcolor=colors['bg'],
        paper_bgcolor=colors['bg'],
        font={'color': colors['text']},
        margin=dict(l=200, r=60, t=80, b=60),
        height=max(400, len(counts) * 40 + 150),
        hovermode='y unified'
    )
    
    return fig


def plot_hourly_pattern(trips: pd.DataFrame, theme: str = "dark") -> go.Figure:
    """Create an improved hourly pattern chart with better styling and insights."""
    df = trips.copy()
    
    if df.empty:
        return _create_empty_figure("Hourly Pickup Pattern", "No trip data available", theme)
        
    if "hour" not in df.columns:
        return _create_empty_figure("Hourly Pickup Pattern", "Hour information not available", theme)
    
    counts = df["hour"].value_counts().sort_index().reset_index()
    counts.columns = ["hour", "trips"]

    # Ensure all 24 hours are present for a consistent look
    all_hours = pd.DataFrame({'hour': range(24)})
    counts = pd.merge(all_hours, counts, on='hour', how='left').fillna(0)
    counts['trips'] = counts['trips'].astype(int)
    
    # Identify peak hours and time periods
    peak_hour = counts.loc[counts['trips'].idxmax(), 'hour'] if counts['trips'].max() > 0 else 12
    
    # Create time period labels
    def get_period(hour):
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 22:
            return "Evening"
        else:
            return "Late Night"
    
    counts['period'] = counts['hour'].apply(get_period)
    counts['hour_label'] = counts['hour'].apply(lambda h: f"{h:02d}:00")
    
    # Color coding based on time periods
    period_colors = {
        "Morning": "#FCD34D",      # Yellow
        "Afternoon": "#F97316",    # Orange  
        "Evening": "#EF4444",      # Red
        "Late Night": "#6366F1"   # Indigo
    }
    
    colors = _get_theme_colors(theme)
    bar_colors = [period_colors[period] for period in counts['period']]

    fig = go.Figure()
    
    # Add bars with period-based coloring
    fig.add_trace(go.Bar(
        x=counts['hour'],
        y=counts['trips'],
        marker=dict(
            color=bar_colors,
            line=dict(color=colors['border'], width=1),
            opacity=0.8
        ),
        text=counts['trips'],
        textposition='auto',
        hovertemplate='<b>%{text} trips at %{customdata}</b><br>' + 
                     'Period: %{meta}<extra></extra>',
        customdata=counts['hour_label'],
        meta=counts['period']
    ))
    
    # Add peak hour annotation
    peak_trips = counts.loc[counts['hour'] == peak_hour, 'trips'].iloc[0]
    if peak_trips > 0:
        fig.add_annotation(
            x=peak_hour,
            y=peak_trips,
            text=f"Peak Hour<br>{peak_hour:02d}:00",
            showarrow=True,
            arrowhead=2,
            arrowcolor=colors['accent'],
            bgcolor=colors['accent'],
            bordercolor=colors['accent'],
            font={'color': 'white', 'size': 12}
        )

    fig.update_layout(
        title={
            'text': "🕐 Hourly Pickup Patterns",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': colors['text']}
        },
        xaxis=dict(
            title=dict(text="Hour of Day", font=dict(color=colors['text'])),
            tickmode='linear',
            dtick=2,  # Show every 2 hours for cleaner display
            range=[-0.5, 23.5],
            gridcolor=colors['grid'],
            tickcolor=colors['text'],
            tickfont=dict(color=colors['text'])
        ),
        yaxis=dict(
            title=dict(text="Number of Trips", font=dict(color=colors['text'])),
            rangemode='tozero',
            gridcolor=colors['grid'],
            tickcolor=colors['text'],
            tickfont=dict(color=colors['text'])
        ),
        plot_bgcolor=colors['bg'],
        paper_bgcolor=colors['bg'],
        font={'color': colors['text']},
        height=450,
        margin=dict(l=60, r=60, t=80, b=60),
        hovermode='x unified'
    )
    
    # Add period legend
    for i, (period, color) in enumerate(period_colors.items()):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=period,
            showlegend=True
        ))
    
    return fig


def make_map_html(
    trips: pd.DataFrame,
    lat_col: str = "dropoff_latitude",
    lon_col: str = "dropoff_longitude",
    max_points: int = 800,
    title: Optional[str] = None,
    theme: str = "dark"
) -> str:
    """Create an improved interactive map with clustering and better styling."""
    df = trips.copy()
    if df.empty or not {lat_col, lon_col}.issubset(df.columns):
        return ""
    
    # Normalize and filter invalid coordinates
    df = df.dropna(subset=[lat_col, lon_col])
    df = _normalize_lat_lon(df, lat_col, lon_col)
    df = df[(df[lat_col].between(29.5, 30.8)) & (df[lon_col].between(-98.2, -97.2))]
    df = df[(df[lat_col] != 0) & (df[lon_col] != 0)]  # Remove zero coordinates
    
    if df.empty:
        return ""
    
    # Sample data if too many points for performance
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    # Use robust center (median); add fallback to mean if NaNs
    center_lat = float(df[lat_col].median()) if pd.notna(df[lat_col].median()) else float(df[lat_col].mean())
    center_lon = float(df[lon_col].median()) if pd.notna(df[lon_col].median()) else float(df[lon_col].mean())

    # Create map with better styling
    map_style = 'CartoDB dark_matter' if theme == 'dark' else 'CartoDB positron'
    
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=12, 
        control_scale=True,
        tiles=map_style
    )
    
    # Add alternative tile layers
    folium.TileLayer('OpenStreetMap').add_to(m)
    if theme == 'dark':
        folium.TileLayer('CartoDB dark_matter').add_to(m)
    else:
        folium.TileLayer('CartoDB positron').add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)

    # Create marker cluster for better performance with many points
    from folium.plugins import MarkerCluster, HeatMap
    
    # Add heatmap layer
    heat_data = [[row[lat_col], row[lon_col]] for _, row in df.iterrows()]
    HeatMap(heat_data, name='Density Heatmap', show=False).add_to(m)
    
    # Add clustered markers
    marker_cluster = MarkerCluster(name='Drop-off Locations', show=True).add_to(m)
    
    # Color scheme based on theme
    colors = _get_theme_colors(theme)
    marker_color = colors['accent']
    
    for idx, row in df.iterrows():
        # Create popup with trip information
        popup_text = f"""
        <b>Drop-off Location</b><br>
        <b>Coordinates:</b> {row[lat_col]:.4f}, {row[lon_col]:.4f}<br>
        """
        
        # Add additional info if available
        if 'dropoff_address' in df.columns and pd.notna(row.get('dropoff_address')):
            popup_text += f"<b>Address:</b> {row['dropoff_address']}<br>"
        if 'num_riders' in df.columns and pd.notna(row.get('num_riders')):
            popup_text += f"<b>Group Size:</b> {row['num_riders']} riders<br>"
        if 'event_time' in df.columns and pd.notna(row.get('event_time')):
            popup_text += f"<b>Time:</b> {row['event_time']}<br>"
        
        folium.CircleMarker(
            location=(row[lat_col], row[lon_col]),
            radius=4,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.7,
            opacity=0.8,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"Drop-off at {row[lat_col]:.3f}, {row[lon_col]:.3f}"
        ).add_to(marker_cluster)

    # Add title if provided
    if title:
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 300px; height: 60px; 
                    background-color: {'rgba(14, 17, 23, 0.8)' if theme == 'dark' else 'rgba(255, 255, 255, 0.8)'};
                    color: {'#FAFAFA' if theme == 'dark' else '#0E1117'};
                    border: 2px solid {colors['accent']};
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 16px;
                    font-weight: bold;
                    text-align: center;
                    z-index: 9999;">
            📍 {title}
            <br><small>{len(df)} locations shown</small>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

    return m._repr_html_()
