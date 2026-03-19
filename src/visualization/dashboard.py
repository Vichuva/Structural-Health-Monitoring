import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.bridges.kaggle_bridge_pipeline import run_bridge_inference, run_kaggle_bridge_pipeline

st.set_page_config(page_title="Bridge SHM Control Room", layout="wide")

registry_path = project_root / "data" / "bridges" / "bridge_registry.csv"
predictions_path = project_root / "data" / "bridges" / "bridge_predictions.csv"
metrics_path = project_root / "models" / "bridge_anomaly_metrics.json"


def inject_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(18, 88, 125, 0.28), transparent 32%),
                radial-gradient(circle at top right, rgba(191, 84, 0, 0.20), transparent 28%),
                linear-gradient(180deg, #07121d 0%, #0d1724 42%, #0b1017 100%);
            color: #e6eef8;
            font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-panel {
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(135, 178, 220, 0.18);
            background: linear-gradient(135deg, rgba(10, 24, 36, 0.94), rgba(14, 36, 52, 0.88));
            border-radius: 22px;
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
            margin-bottom: 1rem;
        }
        .hero-kicker {
            color: #7dd3fc;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.16rem;
            margin-bottom: 0.45rem;
        }
        .hero-title {
            font-size: 2.35rem;
            font-weight: 700;
            color: #f8fbff;
            line-height: 1.05;
            margin-bottom: 0.55rem;
        }
        .hero-copy {
            max-width: 68rem;
            color: #b8cbdf;
            font-size: 1rem;
        }
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0 1.3rem 0;
        }
        .metric-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px solid rgba(129, 167, 205, 0.16);
            background: rgba(10, 20, 30, 0.72);
        }
        .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08rem;
            color: #7f99b4;
        }
        .metric-value {
            color: #f7fbff;
            font-size: 1.55rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }
        .pipeline-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 0.8rem;
            margin-top: 0.8rem;
        }
        .pipeline-card {
            padding: 1rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(9, 20, 30, 0.95), rgba(15, 37, 55, 0.88));
            border: 1px solid rgba(104, 160, 214, 0.18);
            min-height: 160px;
        }
        .pipeline-stage {
            color: #eff7ff;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .pipeline-detail {
            color: #a8c0d8;
            font-size: 0.9rem;
            line-height: 1.45;
            min-height: 4.3rem;
        }
        .pipeline-meta {
            color: #77d5b2;
            font-size: 0.82rem;
            margin-top: 0.7rem;
        }
        .panel-shell {
            padding: 1rem 1.1rem;
            border-radius: 20px;
            border: 1px solid rgba(121, 160, 199, 0.14);
            background: rgba(7, 15, 24, 0.74);
            margin-bottom: 1rem;
        }
        .section-title {
            color: #edf7ff;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_bridge_assets():
    if registry_path.exists() and predictions_path.exists() and metrics_path.exists():
        return
    with st.spinner("Training advanced bridge ensemble and preparing digital twin assets..."):
        run_kaggle_bridge_pipeline()


def load_registry():
    return pd.read_csv(registry_path)


def load_metrics():
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_bridge_summary():
    registry = load_registry()
    predictions = pd.read_csv(predictions_path)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce")
    summary = (
        predictions.groupby("bridge_id")
        .agg(
            anomaly_count=("anomaly", "sum"),
            avg_probability=("anomaly_probability", "mean"),
            max_probability=("anomaly_probability", "max"),
            latest_timestamp=("timestamp", "max"),
        )
        .reset_index()
    )
    merged = registry.merge(summary, on="bridge_id", how="left")
    merged["anomaly_count"] = merged["anomaly_count"].fillna(0).astype(int)
    merged["avg_probability"] = merged["avg_probability"].fillna(0.0)
    merged["max_probability"] = merged["max_probability"].fillna(0.0)
    return merged


def render_header(summary_df, metrics):
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Bridge Digital Twin Command Layer</div>
            <div class="hero-title">Spatiotemporal anomaly orchestration across a live bridge fleet</div>
            <div class="hero-copy">
                The control room fuses bridge telemetry, GNSS, synthetic InSAR deformation frames, and a stacked
                spatiotemporal ensemble model. Select any bridge to replay the pipeline, score the bridge, and project
                anomalies into the 3D twin with hot zones exposed.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bridge_count = int(summary_df["bridge_id"].nunique())
    total_anomalies = int(summary_df["anomaly_count"].sum())
    max_probability = float(summary_df["max_probability"].max())
    model_name = metrics.get("model_name", "advanced_bridge_model")

    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-card">
                <div class="metric-label">Fleet Bridges</div>
                <div class="metric-value">{bridge_count}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Open Anomaly Events</div>
                <div class="metric-value">{total_anomalies}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Peak Fleet Risk</div>
                <div class="metric-value">{max_probability:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Scoring Engine</div>
                <div class="metric-value" style="font-size:1.0rem;">{model_name}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_fleet_map(summary_df, selected_bridge_id):
    marker_sizes = 18 + summary_df["anomaly_count"].clip(lower=1) * 0.45
    marker_sizes = marker_sizes.clip(upper=34)
    marker_colors = summary_df["max_probability"].clip(0, 1)

    figure = go.Figure()

    for _, row in summary_df.iterrows():
        if row["bridge_id"] == selected_bridge_id:
            continue
        selected_row = summary_df.loc[summary_df["bridge_id"] == selected_bridge_id].iloc[0]
        figure.add_trace(
            go.Scattergeo(
                lon=[selected_row["lon"], row["lon"]],
                lat=[selected_row["lat"], row["lat"]],
                mode="lines",
                line={"width": 1.2, "color": "rgba(112, 184, 255, 0.28)"},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    figure.add_trace(
        go.Scattergeo(
            lon=summary_df["lon"],
            lat=summary_df["lat"],
            text=summary_df["bridge_name"],
            customdata=np.stack(
                [
                    summary_df["bridge_id"],
                    summary_df["city"],
                    summary_df["anomaly_count"].astype(str),
                    summary_df["max_probability"].map(lambda value: f"{value:.3f}"),
                ],
                axis=1,
            ),
            mode="markers+text",
            textposition="top center",
            marker={
                "size": marker_sizes,
                "color": marker_colors,
                "colorscale": "Turbo",
                "cmin": 0,
                "cmax": 1,
                "line": {"width": 2.5, "color": "#eff7ff"},
                "opacity": 0.94,
                "colorbar": {"title": "Max Risk"},
            },
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Bridge ID: %{customdata[0]}<br>"
                "City: %{customdata[1]}<br>"
                "Anomalies: %{customdata[2]}<br>"
                "Peak risk: %{customdata[3]}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    figure.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="#0f2435",
        showocean=True,
        oceancolor="#08131f",
        showlakes=True,
        lakecolor="#0a1a2b",
        subunitcolor="rgba(152, 181, 204, 0.35)",
        countrycolor="rgba(152, 181, 204, 0.35)",
    )
    figure.update_layout(
        height=430,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eaf4ff"},
    )
    return figure


def _signal_positions(pred_df):
    ranked = pred_df.sort_values("anomaly_probability", ascending=False).head(180).copy()
    if ranked.empty:
        return ranked, np.array([]), np.array([]), np.array([])

    stress = pd.to_numeric(ranked.get("Simulated_Localized_Stress_Index"), errors="coerce")
    if stress.notna().any():
        span_ratio = (stress - stress.min()) / (stress.max() - stress.min() + 1e-9)
    else:
        span_ratio = pd.Series(np.linspace(0.05, 0.95, len(ranked)), index=ranked.index)

    x_position = 70 + 860 * span_ratio.to_numpy()
    location = ranked.get("Vibration_Anomaly_Location", pd.Series("Deck", index=ranked.index))
    location = location.fillna("Deck").astype(str)

    y_position = np.zeros(len(ranked))
    z_base = np.full(len(ranked), 20.0)

    cable_mask = location.str.contains("Cable", case=False, regex=False)
    pier_mask = location.str.contains("Pier", case=False, regex=False)
    deck_mask = ~(cable_mask | pier_mask)

    y_position[cable_mask.to_numpy()] = np.where(np.arange(cable_mask.sum()) % 2 == 0, -19, 19)
    z_base[cable_mask.to_numpy()] = 52
    x_position[pier_mask.to_numpy()] = np.where(x_position[pier_mask.to_numpy()] < 500, 250, 750)
    z_base[pier_mask.to_numpy()] = 7
    y_position[deck_mask.to_numpy()] = 0

    return ranked, x_position, y_position, z_base


def build_bridge_3d_figure(pred_df, bridge_name):
    span = np.linspace(0, 1000, 280)
    deck_z = 18 + 0.9 * np.sin(span / 65)
    cable_z = 78 - ((span - 500) ** 2) / 7600

    figure = go.Figure()

    figure.add_trace(
        go.Mesh3d(
            x=np.concatenate([span, span]),
            y=np.concatenate([np.full_like(span, -8), np.full_like(span, 8)]),
            z=np.concatenate([deck_z, deck_z]),
            color="#6a7f93",
            opacity=0.48,
            name="Bridge Deck",
            showscale=False,
        )
    )

    for offset, name in [(-22, "Cable South"), (22, "Cable North")]:
        figure.add_trace(
            go.Scatter3d(
                x=span,
                y=np.full_like(span, offset),
                z=cable_z,
                mode="lines",
                line={"width": 4, "color": "#c8d2dd"},
                name=name,
            )
        )

    for tower_x in [250, 750]:
        figure.add_trace(
            go.Scatter3d(
                x=[tower_x, tower_x],
                y=[0, 0],
                z=[0, 86],
                mode="lines",
                line={"width": 10, "color": "#4b5f72"},
                name="Tower" if tower_x == 250 else "Tower ",
                showlegend=(tower_x == 250),
            )
        )

    sensor_nodes = np.linspace(80, 920, 12)
    sensor_heights = np.interp(sensor_nodes, span, deck_z) + 1.2
    figure.add_trace(
        go.Scatter3d(
            x=sensor_nodes,
            y=np.zeros_like(sensor_nodes),
            z=sensor_heights,
            mode="markers",
            marker={"size": 6, "color": "#4dd0e1"},
            name="Sensor Nodes",
        )
    )

    ranked, x_position, y_position, z_base = _signal_positions(pred_df)
    if not ranked.empty:
        z_top = z_base + 10 + 48 * ranked["anomaly_probability"].clip(0, 1).to_numpy()
        beam_x = []
        beam_y = []
        beam_z = []
        for x_value, y_value, z_start, z_end in zip(x_position, y_position, z_base, z_top):
            beam_x.extend([x_value, x_value, None])
            beam_y.extend([y_value, y_value, None])
            beam_z.extend([z_start, z_end, None])

        figure.add_trace(
            go.Scatter3d(
                x=beam_x,
                y=beam_y,
                z=beam_z,
                mode="lines",
                line={"width": 4, "color": "rgba(255, 164, 79, 0.45)"},
                hoverinfo="skip",
                name="Risk Beams",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=x_position,
                y=y_position,
                z=z_top,
                mode="markers",
                marker={
                    "size": 7 + 18 * ranked["anomaly_probability"].clip(0, 1).to_numpy(),
                    "color": ranked["anomaly_probability"],
                    "colorscale": "Turbo",
                    "cmin": 0,
                    "cmax": 1,
                    "showscale": True,
                    "colorbar": {"title": "Risk"},
                    "opacity": 0.96,
                },
                customdata=np.stack(
                    [
                        ranked["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                        ranked["anomaly_probability"].map(lambda value: f"{value:.3f}"),
                        ranked.get("Vibration_Anomaly_Location", pd.Series("Deck", index=ranked.index))
                        .fillna("Deck")
                        .astype(str),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "Timestamp: %{customdata[0]}<br>"
                    "Risk: %{customdata[1]}<br>"
                    "Zone: %{customdata[2]}<br>"
                    "Span X: %{x:.1f} m<extra></extra>"
                ),
                name="Anomaly Markers",
            )
        )

    figure.update_layout(
        title=f"3D Twin Projection - {bridge_name}",
        height=720,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene={
            "xaxis": {"title": "Bridge Span (m)", "backgroundcolor": "rgba(0,0,0,0)"},
            "yaxis": {"title": "Lateral Offset (m)", "backgroundcolor": "rgba(0,0,0,0)"},
            "zaxis": {"title": "Elevation (m)", "backgroundcolor": "rgba(0,0,0,0)"},
            "camera": {"eye": {"x": 1.55, "y": 1.25, "z": 0.85}},
            "aspectratio": {"x": 2.6, "y": 0.7, "z": 0.8},
        },
        font={"color": "#eff7ff"},
    )
    return figure


def build_timeseries_figure(frame, columns, title, colors):
    figure = go.Figure()
    for column, color in zip(columns, colors):
        if column not in frame.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame[column],
                mode="lines",
                name=column,
                line={"width": 2.2, "color": color},
            )
        )
    figure.update_layout(
        title=title,
        height=320,
        margin={"l": 0, "r": 0, "t": 38, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eaf4ff"},
        xaxis={"title": None},
        yaxis={"title": None, "gridcolor": "rgba(158, 193, 220, 0.12)"},
        legend={"orientation": "h"},
    )
    return figure


def animate_runtime_trace(runtime):
    status = st.status(
        f"Running {runtime['model_name']} for {runtime['bridge_id']}...",
        expanded=True,
    )
    progress = st.progress(0)

    stage_count = max(1, len(runtime["stages"]))
    for index, stage in enumerate(runtime["stages"], start=1):
        progress.progress(index / stage_count)
        status.write(
            f"`{stage['stage']}` in {stage['duration_ms']} ms: {stage['detail']}"
        )
        time.sleep(0.08)

    status.update(
        label=f"Pipeline complete in {runtime['total_duration_ms']} ms",
        state="complete",
        expanded=False,
    )


def render_stage_cards(runtime):
    stages = runtime["stages"]
    if not stages:
        return

    columns_per_row = 3
    for row_start in range(0, len(stages), columns_per_row):
        row_stages = stages[row_start : row_start + columns_per_row]
        row_columns = st.columns(columns_per_row, gap="small")

        for index, stage in enumerate(row_stages):
            meta_parts = []
            for key, value in stage.items():
                if key in {"stage", "detail"}:
                    continue
                label = key.replace("_", " ").title()
                meta_parts.append(f"{label}: {value}")

            meta_html = "<br>".join(meta_parts)
            card_html = (
                '<div class="pipeline-card">'
                f'<div class="pipeline-stage">{stage["stage"]}</div>'
                f'<div class="pipeline-detail">{stage["detail"]}</div>'
                f'<div class="pipeline-meta">{meta_html}</div>'
                "</div>"
            )
            row_columns[index].markdown(card_html, unsafe_allow_html=True)


def load_predictions_for_bridge(bridge_id):
    bridge_path = project_root / "data" / "bridges" / bridge_id / "predictions.csv"
    if bridge_path.exists():
        frame = pd.read_csv(bridge_path)
    else:
        frame, _ = run_bridge_inference(bridge_id, return_trace=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


inject_theme()
ensure_bridge_assets()

summary_df = load_bridge_summary()
metrics = load_metrics()

if "selected_bridge_id" not in st.session_state:
    st.session_state.selected_bridge_id = summary_df.iloc[0]["bridge_id"]

render_header(summary_df, metrics)

left, right = st.columns([1.45, 0.85], gap="large")

with left:
    st.markdown('<div class="panel-shell"><div class="section-title">Fleet Risk Map</div>', unsafe_allow_html=True)
    fleet_map = build_fleet_map(summary_df, st.session_state.selected_bridge_id)
    selection_payload = None
    try:
        selection_payload = st.plotly_chart(
            fleet_map,
            use_container_width=True,
            key="fleet_map",
            on_select="rerun",
            selection_mode=("points",),
        )
    except TypeError:
        st.plotly_chart(fleet_map, use_container_width=True, key="fleet_map_static")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel-shell"><div class="section-title">Model Diagnostics</div>', unsafe_allow_html=True)
    st.metric("Model F1", f"{metrics.get('f1', 0):.3f}")
    st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}" if metrics.get("roc_auc") is not None else "N/A")
    st.caption(f"Threshold: {metrics.get('threshold', 0):.4f}")
    st.caption(f"Feature count: {metrics.get('feature_count', 0)}")
    st.markdown("</div>", unsafe_allow_html=True)

if isinstance(selection_payload, dict):
    points = selection_payload.get("selection", {}).get("points", [])
    if points:
        selected_payload = points[0].get("customdata")
        if isinstance(selected_payload, (list, tuple)) and selected_payload:
            st.session_state.selected_bridge_id = selected_payload[0]
        else:
            point_index = int(points[0].get("point_index", 0))
            if 0 <= point_index < len(summary_df):
                st.session_state.selected_bridge_id = summary_df.iloc[point_index]["bridge_id"]

bridge_options = summary_df["bridge_id"].tolist()
selected_index = bridge_options.index(st.session_state.selected_bridge_id)
selected_bridge_id = st.selectbox(
    "Selected bridge",
    options=bridge_options,
    index=selected_index,
    format_func=lambda bridge_id: summary_df.loc[
        summary_df["bridge_id"] == bridge_id, "bridge_name"
    ].iloc[0],
)
st.session_state.selected_bridge_id = selected_bridge_id

selected_row = summary_df.loc[summary_df["bridge_id"] == selected_bridge_id].iloc[0]
st.caption(
    f"Bridge focus: {selected_row['bridge_name']} | {selected_row['city']} | "
    f"peak probability {selected_row['max_probability']:.3f}"
)

bridge_changed = st.session_state.get("_last_loaded_bridge") != selected_bridge_id
rerun_requested = st.button("Replay processing pipeline", type="primary")

if bridge_changed or rerun_requested:
    pred_df, runtime = run_bridge_inference(selected_bridge_id, return_trace=True)
    st.session_state["_last_loaded_bridge"] = selected_bridge_id
    st.session_state["_last_runtime"] = runtime
    st.session_state["_last_pred_df"] = pred_df
    animate_runtime_trace(runtime)
else:
    runtime = st.session_state.get("_last_runtime")
    pred_df = st.session_state.get("_last_pred_df")
    if pred_df is None or runtime is None:
        pred_df, runtime = run_bridge_inference(selected_bridge_id, return_trace=True)
        st.session_state["_last_runtime"] = runtime
        st.session_state["_last_pred_df"] = pred_df

pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce")
pred_df = pred_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

st.markdown('<div class="panel-shell"><div class="section-title">Runtime Pipeline</div>', unsafe_allow_html=True)
render_stage_cards(runtime)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="panel-shell"><div class="section-title">3D Digital Twin</div>', unsafe_allow_html=True)
st.plotly_chart(
    build_bridge_3d_figure(pred_df, selected_row["bridge_name"]),
    use_container_width=True,
)
st.markdown("</div>", unsafe_allow_html=True)

bridge_dir = project_root / "data" / "bridges" / selected_bridge_id
gnss_file = bridge_dir / "gnss_raw.csv"
insar_file = bridge_dir / "insar_timeseries.csv"
sensor_file = bridge_dir / "sensor_data.csv"

tabs = st.tabs(["Telemetry", "InSAR Vision", "Event Ledger"])

with tabs[0]:
    telemetry_left, telemetry_right = st.columns(2, gap="large")

    if gnss_file.exists():
        gnss = pd.read_csv(gnss_file)
        gnss["timestamp"] = pd.to_datetime(gnss["timestamp"], errors="coerce")
        gnss = gnss.dropna(subset=["timestamp"])
        telemetry_left.plotly_chart(
            build_timeseries_figure(
                gnss,
                ["x", "y", "z"],
                "GNSS Coordinate Stream",
                ["#4dd0e1", "#90caf9", "#ffb86b"],
            ),
            use_container_width=True,
        )

    if insar_file.exists():
        insar = pd.read_csv(insar_file)
        insar["timestamp"] = pd.to_datetime(insar["timestamp"], errors="coerce")
        insar = insar.dropna(subset=["timestamp"])
        telemetry_right.plotly_chart(
            build_timeseries_figure(
                insar,
                ["los_displacement"],
                "InSAR LOS Deformation",
                ["#ff8a65"],
            ),
            use_container_width=True,
        )

    if sensor_file.exists():
        sensor = pd.read_csv(sensor_file)
        sensor["timestamp"] = pd.to_datetime(sensor["timestamp"], errors="coerce")
        sensor = sensor.dropna(subset=["timestamp"])
        sensor_columns = [
            column
            for column in [
                "Strain_microstrain",
                "Deflection_mm",
                "Vibration_ms2",
                "Tilt_deg",
                "Temperature_C",
                "Humidity_percent",
            ]
            if column in sensor.columns
        ]
        if sensor_columns:
            st.plotly_chart(
                build_timeseries_figure(
                    sensor,
                    sensor_columns,
                    "Sensor Fusion Panel",
                    ["#80cbc4", "#ffd54f", "#ef9a9a", "#b39ddb", "#81d4fa", "#ffcc80"],
                ),
                use_container_width=True,
            )

with tabs[1]:
    mask_meta_file = bridge_dir / "insar_mask_metadata.csv"
    if mask_meta_file.exists():
        mask_meta = pd.read_csv(mask_meta_file)
        if not mask_meta.empty:
            mask_meta["timestamp"] = pd.to_datetime(mask_meta["timestamp"], errors="coerce")
            mask_meta = mask_meta.sort_values("timestamp", na_position="last").reset_index(drop=True)
            frame_index = st.slider("InSAR frame", 0, len(mask_meta) - 1, 0)
            frame = mask_meta.iloc[frame_index]

            image_col, mask_col, overlay_col = st.columns(3)
            image_col.image(frame["image_path"], caption="SAR intensity frame", use_container_width=True)
            mask_col.image(frame["mask_path"], caption="Predicted deformation mask", use_container_width=True)
            overlay_col.image(frame["overlay_path"], caption="Mask composited over SAR", use_container_width=True)

            st.caption(
                f"Frame timestamp: {pd.to_datetime(frame['timestamp'])} | "
                f"masked area ratio: {float(frame['mask_ratio']):.4f}"
            )
    else:
        st.info("No InSAR image frames are available for the selected bridge.")

with tabs[2]:
    anomalies = pred_df[pred_df["anomaly"] == 1].copy()
    if anomalies.empty:
        st.write("No anomaly rows exceeded the runtime threshold for this bridge.")
    else:
        show_columns = [
            "timestamp",
            "anomaly_probability",
            "Vibration_Anomaly_Location",
            "Localized_Strain_Hotspot",
            "Deflection_mm",
            "Displacement_mm",
            "Vibration_ms2",
            "Probability_of_Failure_PoF",
            "Structural_Health_Index_SHI",
        ]
        show_columns = [column for column in show_columns if column in anomalies.columns]
        st.dataframe(
            anomalies[show_columns].sort_values("anomaly_probability", ascending=False).head(350),
            use_container_width=True,
        )
