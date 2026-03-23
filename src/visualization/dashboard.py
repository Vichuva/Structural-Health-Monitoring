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
xai_summary_path = project_root / "models" / "bridge_xai_summary.json"


def inject_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(18,88,125,0.28), transparent 32%),
                radial-gradient(circle at top right, rgba(191,84,0,0.20), transparent 28%),
                linear-gradient(180deg, #07121d 0%, #0d1724 42%, #0b1017 100%);
            color: #e6eef8;
            font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        }
        .block-container { max-width: 1480px; padding-top: 1.5rem; padding-bottom: 2rem; }
        .hero, .metric-card, .pipe-card {
            border: 1px solid rgba(121,160,199,0.14);
            background: rgba(7,15,24,0.78);
            border-radius: 22px;
            box-shadow: 0 16px 60px rgba(0,0,0,0.22);
        }
        .hero { padding: 1.35rem 1.5rem; margin-bottom: 1rem; }
        .kicker { color: #7dd3fc; font-size: .82rem; text-transform: uppercase; letter-spacing: .16rem; }
        .title { color: #f8fbff; font-size: 2.2rem; font-weight: 700; line-height: 1.04; margin-top: .25rem; }
        .copy { color: #b8cbdf; font-size: 1rem; margin-top: .45rem; }
        .metric-card { padding: 1rem 1.1rem; min-height: 120px; }
        .metric-label { color: #7f99b4; font-size: .8rem; text-transform: uppercase; letter-spacing: .08rem; }
        .metric-value { color: #f7fbff; font-size: 1.65rem; font-weight: 700; margin-top: .3rem; }
        .metric-copy { color: #9fb8ce; font-size: .88rem; margin-top: .35rem; line-height: 1.35; }
        .pipe-card { padding: 1rem; min-height: 190px; }
        .pipe-stage { color: #eff7ff; font-size: 1.02rem; font-weight: 700; margin-bottom: .35rem; }
        .pipe-detail { color: #a8c0d8; font-size: .91rem; line-height: 1.48; min-height: 4.6rem; }
        .pipe-meta { color: #77d5b2; font-size: .82rem; margin-top: .7rem; line-height: 1.45; }
        .stTabs [data-baseweb="tab-list"] { gap: .5rem; }
        .stTabs [data-baseweb="tab"] { border-radius: 999px; padding: .45rem 1rem; background: rgba(19,34,48,.9); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_assets():
    if registry_path.exists() and predictions_path.exists() and metrics_path.exists():
        return
    with st.spinner("Preparing bridge analytics assets..."):
        run_kaggle_bridge_pipeline()


def load_summary():
    registry = pd.read_csv(registry_path)
    preds = pd.read_csv(predictions_path)
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], errors="coerce")
    summary = (
        preds.groupby("bridge_id")
        .agg(
            anomaly_count=("anomaly", "sum"),
            avg_probability=("anomaly_probability", "mean"),
            max_probability=("anomaly_probability", "max"),
        )
        .reset_index()
    )
    merged = registry.merge(summary, on="bridge_id", how="left")
    merged["anomaly_count"] = merged["anomaly_count"].fillna(0).astype(int)
    merged["avg_probability"] = merged["avg_probability"].fillna(0.0)
    merged["max_probability"] = merged["max_probability"].fillna(0.0)
    return merged


def metric_card(column, label, value, copy):
    column.markdown(
        (
            '<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-copy">{copy}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def section(title):
    st.subheader(title)


def build_map(summary, selected_bridge_id):
    figure = go.Figure()
    current = summary.loc[summary["bridge_id"] == selected_bridge_id].iloc[0]
    for _, row in summary.iterrows():
        if row["bridge_id"] == selected_bridge_id:
            continue
        figure.add_trace(
            go.Scattergeo(
                lon=[current["lon"], row["lon"]],
                lat=[current["lat"], row["lat"]],
                mode="lines",
                line={"width": 1.0, "color": "rgba(110,193,255,0.22)"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scattergeo(
            lon=summary["lon"],
            lat=summary["lat"],
            text=summary["bridge_name"],
            customdata=np.stack(
                [
                    summary["bridge_id"],
                    summary["city"],
                    summary["region"],
                    summary["anomaly_count"].astype(str),
                    summary["max_probability"].map(lambda v: f"{v:.3f}"),
                ],
                axis=1,
            ),
            mode="markers+text",
            textposition="top center",
            marker={
                "size": (18 + summary["anomaly_count"].clip(lower=1) * 0.5).clip(upper=34),
                "color": summary["max_probability"],
                "colorscale": "Turbo",
                "cmin": 0,
                "cmax": 1,
                "line": {"width": 2.4, "color": "#eff7ff"},
                "opacity": 0.95,
                "colorbar": {"title": "Max Risk"},
            },
            hovertemplate=(
                "<b>%{text}</b><br>Bridge ID: %{customdata[0]}<br>City: %{customdata[1]}"
                "<br>Region: %{customdata[2]}<br>Anomalies: %{customdata[3]}"
                "<br>Peak risk: %{customdata[4]}<extra></extra>"
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
        subunitcolor="rgba(152,181,204,0.35)",
        countrycolor="rgba(152,181,204,0.35)",
    )
    figure.update_layout(
        height=450,
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
        ratio = (stress - stress.min()) / (stress.max() - stress.min() + 1e-9)
    else:
        ratio = pd.Series(np.linspace(0.05, 0.95, len(ranked)), index=ranked.index)
    x_pos = 70 + 860 * ratio.to_numpy()
    loc = ranked.get("Vibration_Anomaly_Location", pd.Series("Deck", index=ranked.index)).fillna("Deck").astype(str)
    y_pos = np.zeros(len(ranked))
    z_base = np.full(len(ranked), 20.0)
    cable_mask = loc.str.contains("Cable", case=False, regex=False)
    pier_mask = loc.str.contains("Pier", case=False, regex=False)
    y_pos[cable_mask.to_numpy()] = np.where(np.arange(cable_mask.sum()) % 2 == 0, -19, 19)
    z_base[cable_mask.to_numpy()] = 52
    x_pos[pier_mask.to_numpy()] = np.where(x_pos[pier_mask.to_numpy()] < 500, 250, 750)
    z_base[pier_mask.to_numpy()] = 7
    return ranked, x_pos, y_pos, z_base


def build_bridge_figure(pred_df, bridge_name):
    span = np.linspace(0, 1000, 280)
    deck_z = 18 + 0.9 * np.sin(span / 65)
    cable_z = 78 - ((span - 500) ** 2) / 7600
    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=np.concatenate([span, span]),
            y=np.concatenate([np.full_like(span, -8), np.full_like(span, 8)]),
            z=np.concatenate([deck_z, deck_z]),
            color="#61788b",
            opacity=0.52,
            name="Deck",
            showscale=False,
        )
    )
    for offset, name in [(-22, "Cable South"), (22, "Cable North")]:
        figure.add_trace(
            go.Scatter3d(x=span, y=np.full_like(span, offset), z=cable_z, mode="lines", line={"width": 4, "color": "#d0d9e2"}, name=name)
        )
    for tower_x in [250, 750]:
        figure.add_trace(
            go.Scatter3d(
                x=[tower_x, tower_x], y=[0, 0], z=[0, 86], mode="lines",
                line={"width": 10, "color": "#4b5f72"}, name="Tower" if tower_x == 250 else "Tower ", showlegend=(tower_x == 250)
            )
        )
    nodes = np.linspace(80, 920, 12)
    figure.add_trace(
        go.Scatter3d(x=nodes, y=np.zeros_like(nodes), z=np.interp(nodes, span, deck_z) + 1.2, mode="markers", marker={"size": 6, "color": "#4dd0e1"}, name="Sensor Nodes")
    )
    ranked, x_pos, y_pos, z_base = _signal_positions(pred_df)
    if not ranked.empty:
        z_top = z_base + 10 + 48 * ranked["anomaly_probability"].clip(0, 1).to_numpy()
        bx, by, bz = [], [], []
        for xv, yv, zs, ze in zip(x_pos, y_pos, z_base, z_top):
            bx.extend([xv, xv, None]); by.extend([yv, yv, None]); bz.extend([zs, ze, None])
        figure.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="lines", line={"width": 4, "color": "rgba(255,164,79,0.45)"}, hoverinfo="skip", name="Risk Beams"))
        figure.add_trace(
            go.Scatter3d(
                x=x_pos, y=y_pos, z=z_top, mode="markers",
                marker={"size": 7 + 18 * ranked["anomaly_probability"].clip(0, 1).to_numpy(), "color": ranked["anomaly_probability"], "colorscale": "Turbo", "cmin": 0, "cmax": 1, "showscale": True, "colorbar": {"title": "Risk"}, "opacity": 0.96},
                customdata=np.stack([ranked["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"), ranked["anomaly_probability"].map(lambda v: f"{v:.3f}")], axis=1),
                hovertemplate="Timestamp: %{customdata[0]}<br>Risk: %{customdata[1]}<br>Span X: %{x:.1f} m<extra></extra>",
                name="Anomaly Markers",
            )
        )
    figure.update_layout(
        title=f"3D Twin Projection - {bridge_name}",
        height=720,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene={"xaxis": {"title": "Bridge Span (m)"}, "yaxis": {"title": "Lateral Offset (m)"}, "zaxis": {"title": "Elevation (m)"}, "camera": {"eye": {"x": 1.55, "y": 1.25, "z": 0.85}}, "aspectratio": {"x": 2.6, "y": 0.7, "z": 0.8}},
        font={"color": "#eff7ff"},
    )
    return figure


def prepare_gnss_demo_frame(frame):
    required = {"timestamp", "x", "y", "z"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    gnss = frame[["timestamp", "x", "y", "z"]].copy()
    gnss["timestamp"] = pd.to_datetime(gnss["timestamp"], errors="coerce")
    gnss = gnss.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if gnss.empty:
        return gnss
    baseline = gnss.loc[0, ["x", "y", "z"]]
    for axis in ["x", "y", "z"]:
        gnss[f"{axis}_mm"] = (pd.to_numeric(gnss[axis], errors="coerce") - float(baseline[axis])) * 1000.0
    gnss = gnss.dropna(subset=["x_mm", "y_mm", "z_mm"]).reset_index(drop=True)
    gnss["total_mm"] = np.sqrt(gnss["x_mm"] ** 2 + gnss["y_mm"] ** 2 + gnss["z_mm"] ** 2)
    return gnss


def line_chart(frame, columns, title, colors, selected_timestamp=None, yaxis_title=None):
    figure = go.Figure()
    for column, color in zip(columns, colors):
        if column in frame.columns:
            figure.add_trace(go.Scatter(x=frame["timestamp"], y=frame[column], mode="lines", name=column, line={"width": 2.2, "color": color}))
    if selected_timestamp is not None:
        figure.add_vline(x=selected_timestamp, line_width=1.4, line_dash="dash", line_color="rgba(255,255,255,0.35)")
    figure.update_layout(
        title=title, height=320, margin={"l": 0, "r": 0, "t": 38, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#eaf4ff"},
        xaxis={"title": None},
        yaxis={"title": yaxis_title, "gridcolor": "rgba(158,193,220,0.12)"},
        legend={"orientation": "h"},
        hovermode="x unified",
    )
    return figure


def bar_chart(frame, x_col, y_col, title, color):
    fig = go.Figure(go.Bar(x=frame[x_col], y=frame[y_col], orientation="h", marker={"color": color}))
    fig.update_layout(
        title=title, height=360, margin={"l": 0, "r": 0, "t": 38, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#eaf4ff"},
        xaxis={"gridcolor": "rgba(158,193,220,0.12)"}, yaxis={"gridcolor": "rgba(158,193,220,0.08)"},
    )
    return fig


def insar_activity(mask_meta, insar_df, selected_timestamp):
    fig = go.Figure()
    if not insar_df.empty and "los_displacement" in insar_df.columns:
        fig.add_trace(go.Scatter(x=insar_df["timestamp"], y=insar_df["los_displacement"], mode="lines", name="LOS displacement", line={"color": "#ff8a65", "width": 2.6}, yaxis="y1"))
    if not mask_meta.empty:
        if "mask_ratio" in mask_meta.columns:
            fig.add_trace(go.Scatter(x=mask_meta["timestamp"], y=mask_meta["mask_ratio"], mode="lines+markers", name="Mask ratio", line={"color": "#6ee7b7", "width": 2.0}, marker={"size": 6}, yaxis="y2"))
        if "coherence_mean" in mask_meta.columns:
            fig.add_trace(go.Scatter(x=mask_meta["timestamp"], y=mask_meta["coherence_mean"], mode="lines", name="Mean coherence", line={"color": "#7dd3fc", "width": 1.8, "dash": "dot"}, yaxis="y2"))
    if selected_timestamp is not None:
        fig.add_vline(x=selected_timestamp, line_width=1.5, line_dash="dash", line_color="rgba(255,255,255,0.4)")
    fig.update_layout(
        title="InSAR temporal activity", height=360, margin={"l": 0, "r": 0, "t": 38, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#eaf4ff"},
        xaxis={"title": None}, yaxis={"title": "LOS disp", "gridcolor": "rgba(158,193,220,0.12)"},
        yaxis2={"title": "Mask / coherence", "overlaying": "y", "side": "right", "showgrid": False},
        legend={"orientation": "h"},
    )
    return fig


def animate_runtime(runtime):
    status = st.status(f"Running {runtime['model_name']} for {runtime['bridge_id']}...", expanded=True)
    progress = st.progress(0)
    total = max(1, len(runtime["stages"]))
    for idx, stage in enumerate(runtime["stages"], start=1):
        progress.progress(idx / total)
        status.write(f"`{stage['stage']}` in {stage['duration_ms']} ms: {stage['detail']}")
        time.sleep(0.08)
    status.update(label=f"Pipeline complete in {runtime['total_duration_ms']} ms", state="complete", expanded=False)


def render_pipeline(runtime):
    stages = runtime.get("stages", [])
    for start in range(0, len(stages), 3):
        cols = st.columns(3, gap="small")
        for idx, stage in enumerate(stages[start : start + 3]):
            meta = []
            for key, value in stage.items():
                if key not in {"stage", "detail"}:
                    meta.append(f"{key.replace('_', ' ').title()}: {value}")
            cols[idx].markdown(
                (
                    '<div class="pipe-card">'
                    f'<div class="pipe-stage">{stage["stage"]}</div>'
                    f'<div class="pipe-detail">{stage["detail"]}</div>'
                    f'<div class="pipe-meta">{"<br>".join(meta)}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


inject_theme()
ensure_assets()
summary = load_summary()
metrics = load_json(metrics_path)
xai_summary = load_json(xai_summary_path)

if "selected_bridge_id" not in st.session_state:
    st.session_state.selected_bridge_id = summary.iloc[0]["bridge_id"]

st.markdown(
    """
    <div class="hero">
        <div class="kicker">Bridge Digital Twin Command Layer</div>
        <div class="title">Explainable multimodal anomaly intelligence for bridge fleets</div>
        <div class="copy">
            Select a bridge to replay the inference pipeline, inspect local anomaly drivers, and analyze deformation
            behavior through advanced InSAR products including interferograms, heatmaps, and coherence maps.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

cards = st.columns(4, gap="small")
metric_card(cards[0], "Fleet Bridges", int(summary["bridge_id"].nunique()), "Registered bridge twins on the fleet map.")
metric_card(cards[1], "Open Anomalies", int(summary["anomaly_count"].sum()), "Events above the calibrated threshold.")
metric_card(cards[2], "Peak Fleet Risk", f"{float(summary['max_probability'].max()):.3f}", "Highest anomaly probability in the fleet.")
metric_card(cards[3], "Scoring Engine", metrics.get("model_name", "ensemble"), "Stacked spatiotemporal ensemble with XAI.")

top_left, top_right = st.columns([1.45, 0.85], gap="large")
with top_left:
    section("Fleet Risk Map")
    selection = None
    try:
        selection = st.plotly_chart(build_map(summary, st.session_state.selected_bridge_id), use_container_width=True, key="fleet_map", on_select="rerun", selection_mode=("points",))
    except TypeError:
        st.plotly_chart(build_map(summary, st.session_state.selected_bridge_id), use_container_width=True, key="fleet_map_static")
with top_right:
    section("Model Diagnostics")
    d1 = st.columns(2, gap="small")
    metric_card(d1[0], "F1", f"{metrics.get('f1', 0):.3f}", "Balanced anomaly retrieval performance.")
    metric_card(d1[1], "Precision", f"{metrics.get('precision', 0):.3f}", "Correctness of anomaly alerts.")
    d2 = st.columns(2, gap="small")
    metric_card(d2[0], "Recall", f"{metrics.get('recall', 0):.3f}", "Share of true anomalies recovered.")
    metric_card(d2[1], "ROC AUC", f"{metrics.get('roc_auc', 0):.3f}" if metrics.get("roc_auc") is not None else "N/A", "Ranking quality between normal and abnormal states.")
    st.caption(f"Threshold: {metrics.get('threshold', 0):.4f} | Feature count: {metrics.get('feature_count', 0)}")

if isinstance(selection, dict):
    points = selection.get("selection", {}).get("points", [])
    if points:
        payload = points[0].get("customdata")
        if isinstance(payload, (list, tuple)) and payload:
            st.session_state.selected_bridge_id = payload[0]

bridge_options = summary["bridge_id"].tolist()
selected_index = bridge_options.index(st.session_state.selected_bridge_id)
bridge_id = st.selectbox(
    "Selected bridge",
    options=bridge_options,
    index=selected_index,
    format_func=lambda value: summary.loc[summary["bridge_id"] == value, "bridge_name"].iloc[0],
)
st.session_state.selected_bridge_id = bridge_id
selected = summary.loc[summary["bridge_id"] == bridge_id].iloc[0]
st.caption(f"Bridge focus: {selected['bridge_name']} | {selected['city']} | {selected['region']} | peak probability {selected['max_probability']:.3f}")

bridge_changed = st.session_state.get("_last_loaded_bridge") != bridge_id
rerun = st.button("Replay processing pipeline", type="primary")
if bridge_changed or rerun:
    pred_df, runtime = run_bridge_inference(bridge_id, return_trace=True)
    st.session_state["_last_loaded_bridge"] = bridge_id
    st.session_state["_last_runtime"] = runtime
    st.session_state["_last_pred_df"] = pred_df
    animate_runtime(runtime)
else:
    runtime = st.session_state.get("_last_runtime")
    pred_df = st.session_state.get("_last_pred_df")
    if pred_df is None or runtime is None:
        pred_df, runtime = run_bridge_inference(bridge_id, return_trace=True)
        st.session_state["_last_runtime"] = runtime
        st.session_state["_last_pred_df"] = pred_df

pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce")
pred_df = pred_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

section("Runtime Pipeline")
render_pipeline(runtime)

section("3D Digital Twin")
st.plotly_chart(build_bridge_figure(pred_df, selected["bridge_name"]), use_container_width=True)

bridge_dir = project_root / "data" / "bridges" / bridge_id
gnss_file = bridge_dir / "gnss_raw.csv"
insar_file = bridge_dir / "insar_timeseries.csv"
sensor_file = bridge_dir / "sensor_data.csv"
mask_meta_file = bridge_dir / "insar_mask_metadata.csv"
local_xai_file = bridge_dir / "xai_top_factors.csv"

tabs = st.tabs(["Telemetry", "InSAR Intelligence", "Explainability", "Event Ledger"])

with tabs[0]:
    row = st.columns(2, gap="large")
    if gnss_file.exists():
        gnss = pd.read_csv(gnss_file)
        gnss_demo = prepare_gnss_demo_frame(gnss)
        if gnss_demo.empty:
            row[0].info("GNSS telemetry is unavailable for the selected bridge.")
        else:
            gnss_stats = row[0].columns(3, gap="small")
            gnss_stats[0].metric("Samples", f"{len(gnss_demo):,}")
            gnss_stats[1].metric("Peak 3D Drift", f"{gnss_demo['total_mm'].max():.1f} mm")
            gnss_stats[2].metric("Vertical Span", f"{gnss_demo['z_mm'].max() - gnss_demo['z_mm'].min():.1f} mm")
            row[0].plotly_chart(
                line_chart(
                    gnss_demo,
                    ["x_mm", "y_mm", "z_mm"],
                    "GNSS displacement stream",
                    ["#4dd0e1", "#90caf9", "#ffb86b"],
                    yaxis_title="Displacement (mm)",
                ),
                use_container_width=True,
            )
            row[0].caption("Offsets are relative to the first GNSS sample so micro-movements remain visible during the demo.")
    else:
        row[0].info("GNSS telemetry file is missing for the selected bridge.")
    if insar_file.exists():
        insar = pd.read_csv(insar_file)
        insar["timestamp"] = pd.to_datetime(insar["timestamp"], errors="coerce")
        insar = insar.dropna(subset=["timestamp"])
        row[1].plotly_chart(line_chart(insar, ["los_displacement"], "InSAR LOS deformation", ["#ff8a65"]), use_container_width=True)
    else:
        row[1].info("InSAR timeseries file is missing for the selected bridge.")
    if sensor_file.exists():
        sensor = pd.read_csv(sensor_file)
        sensor["timestamp"] = pd.to_datetime(sensor["timestamp"], errors="coerce")
        sensor = sensor.dropna(subset=["timestamp"])
        cols = [c for c in ["Strain_microstrain", "Deflection_mm", "Vibration_ms2", "Tilt_deg", "Temperature_C", "Humidity_percent"] if c in sensor.columns]
        if cols:
            st.plotly_chart(line_chart(sensor, cols, "Sensor fusion panel", ["#80cbc4", "#ffd54f", "#ef9a9a", "#b39ddb", "#81d4fa", "#ffcc80"]), use_container_width=True)
        else:
            st.info("Sensor telemetry columns are not available for the selected bridge.")
    else:
        st.info("Sensor telemetry file is missing for the selected bridge.")

with tabs[1]:
    insar = pd.DataFrame()
    if insar_file.exists():
        insar = pd.read_csv(insar_file)
        insar["timestamp"] = pd.to_datetime(insar["timestamp"], errors="coerce")
        insar = insar.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    mask_meta = pd.DataFrame()
    if mask_meta_file.exists():
        mask_meta = pd.read_csv(mask_meta_file)
        if not mask_meta.empty:
            mask_meta["timestamp"] = pd.to_datetime(mask_meta["timestamp"], errors="coerce")
            mask_meta = mask_meta.sort_values("timestamp", na_position="last").reset_index(drop=True)
    if mask_meta.empty:
        st.info("No InSAR products are available for the selected bridge.")
    else:
        frame_idx = st.slider("InSAR frame", 0, len(mask_meta) - 1, 0)
        frame = mask_meta.iloc[frame_idx]
        selected_ts = pd.to_datetime(frame["timestamp"]) if pd.notna(frame["timestamp"]) else None
        top = st.columns([1.4, 0.6], gap="large")
        top[0].plotly_chart(insar_activity(mask_meta, insar, selected_ts), use_container_width=True)
        metric_card(top[1], "Mask Ratio", f"{float(frame['mask_ratio']):.4f}", "Fraction of active deformation pixels.")
        metric_card(top[1], "Deformation Energy", f"{float(frame.get('deformation_energy', 0.0)):.4f}", "Normalized deformation intensity.")
        metric_card(top[1], "Mean Coherence", f"{float(frame.get('coherence_mean', 0.0)):.4f}", "Phase stability after extraction.")
        img_a = st.columns(2, gap="small")
        img_b = st.columns(2, gap="small")
        img_a[0].image(frame["image_path"], caption="Amplitude image", use_container_width=True)
        img_a[1].image(frame["interferogram_path"], caption="Interferogram", use_container_width=True)
        img_b[0].image(frame["heatmap_path"], caption="Deformation heatmap", use_container_width=True)
        img_b[1].image(frame["coherence_path"], caption="Coherence map", use_container_width=True)
        st.image(frame["overlay_path"], caption="Segmented anomaly overlay", use_container_width=True)

with tabs[2]:
    global_xai = pd.DataFrame(xai_summary.get("global_feature_importance", []))
    local_xai = pd.read_csv(local_xai_file) if local_xai_file.exists() else pd.DataFrame()
    cols = st.columns(2, gap="large")
    if not global_xai.empty:
        cols[0].plotly_chart(bar_chart(global_xai.sort_values("importance_mean", ascending=True), "importance_mean", "feature", "Global feature importance", "#7dd3fc"), use_container_width=True)
    else:
        cols[0].info("Global explainability summary is not available.")
    if not local_xai.empty:
        local_xai = local_xai.copy()
        local_xai["signed_feature"] = local_xai["feature"] + local_xai["impact"].map(lambda v: "  (+)" if v >= 0 else "  (-)")
        cols[1].plotly_chart(bar_chart(local_xai.sort_values("impact", ascending=True), "impact", "signed_feature", "Local anomaly drivers", "#ff9f68"), use_container_width=True)
        st.caption("Local explanation uses counterfactual feature ablation. Each bar shows how the top anomaly probability changes when that feature is replaced by the fleet reference value.")
    else:
        cols[1].info("Local explanation is not available for the selected bridge.")

with tabs[3]:
    anomalies = pred_df[pred_df["anomaly"] == 1].copy()
    if anomalies.empty:
        st.write("No anomaly rows exceeded the runtime threshold for this bridge.")
    else:
        display_cols = [c for c in ["timestamp", "anomaly_probability", "Vibration_Anomaly_Location", "Localized_Strain_Hotspot", "Deflection_mm", "Displacement_mm", "Vibration_ms2", "Probability_of_Failure_PoF", "Structural_Health_Index_SHI"] if c in anomalies.columns]
        st.dataframe(anomalies[display_cols].sort_values("anomaly_probability", ascending=False).head(350), use_container_width=True)
