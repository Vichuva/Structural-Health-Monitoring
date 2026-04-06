from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
DATA_ROOT = PROJECT_ROOT / "data"
REPORTS_ROOT = PROJECT_ROOT / "reports"
MODELS_ROOT = PROJECT_ROOT / "models"
CACHE_ROOT = PROJECT_ROOT / ".cache"
MPL_CACHE_ROOT = CACHE_ROOT / "matplotlib"

CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MPL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_ROOT)
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*", category=UserWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in matmul", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in matmul", category=RuntimeWarning)

from main import run_pipeline
from src.bridges.kaggle_bridge_pipeline import run_bridge_inference, run_kaggle_bridge_pipeline
from src.utils.config import BRIDGES_DIR, BRIDGE_MODEL_METRICS_PATH, BRIDGE_PREDICTIONS_PATH, BRIDGE_REGISTRY_PATH

PATH_COLUMNS = [
    "image_path",
    "mask_path",
    "overlay_path",
    "interferogram_path",
    "heatmap_path",
    "coherence_path",
]

HOTSPOT_ZONE_LOOKUP = {
    "Deck": {"label": "Deck", "x": 50, "y": 70},
    "Tower": {"label": "Tower", "x": 35, "y": 30},
    "Cable": {"label": "Cable", "x": 67, "y": 38},
    "Pier": {"label": "Pier", "x": 52, "y": 88},
    "Joint": {"label": "Joint", "x": 24, "y": 74},
}

OPERATIONS: dict[str, dict[str, Any]] = {}
OPERATIONS_LOCK = threading.Lock()


def ensure_core_artifacts() -> None:
    required = [BRIDGE_REGISTRY_PATH, BRIDGE_PREDICTIONS_PATH, BRIDGE_MODEL_METRICS_PATH]
    if all(path.exists() for path in required):
        return
    run_kaggle_bridge_pipeline()


def _progress_step(stage: str, detail: str, status: str = "completed", meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "stage": stage,
        "detail": detail,
        "status": status,
        "meta": meta or {},
        "timestamp": datetime.utcnow().isoformat(),
    }


def _set_operation(operation_id: str, **updates: Any) -> dict[str, Any]:
    with OPERATIONS_LOCK:
        operation = OPERATIONS[operation_id]
        operation.update(updates)
        operation["updated_at"] = datetime.utcnow().isoformat()
        return dict(operation)


def _append_operation_step(operation_id: str, step: dict[str, Any]) -> None:
    with OPERATIONS_LOCK:
        operation = OPERATIONS[operation_id]
        operation["steps"].append(step)
        operation["updated_at"] = datetime.utcnow().isoformat()


def _create_operation(kind: str, target: str | None = None) -> dict[str, Any]:
    operation_id = uuid.uuid4().hex
    operation = {
        "id": operation_id,
        "kind": kind,
        "target": target,
        "status": "running",
        "message": "Queued operation.",
        "progress": 0,
        "steps": [],
        "result": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    with OPERATIONS_LOCK:
        OPERATIONS[operation_id] = operation
    return dict(operation)


def get_operation(operation_id: str) -> dict[str, Any]:
    with OPERATIONS_LOCK:
        if operation_id not in OPERATIONS:
            raise FileNotFoundError(f"Unknown operation: {operation_id}")
        return dict(OPERATIONS[operation_id])


def _run_operation_in_thread(operation_id: str, worker) -> None:
    def runner() -> None:
        try:
            worker()
        except Exception as exc:  # pragma: no cover - surfaced to API caller
            _set_operation(
                operation_id,
                status="failed",
                message=str(exc),
                error=str(exc),
                progress=100,
            )

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any, digits: int | None = None) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if digits is not None:
        return round(result, digits)
    return result


def _safe_int(value: Any) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _iso_timestamp(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.isoformat()


def _resolve_project_asset_path(raw_path: Any) -> Path | None:
    if raw_path is None or (isinstance(raw_path, float) and math.isnan(raw_path)):
        return None

    raw_text = str(raw_path)
    candidate = Path(raw_text)
    if candidate.exists():
        return candidate

    normalized = raw_text.replace("\\", "/")
    marker = "/data/"
    if marker in normalized:
        relative_part = normalized.split(marker, 1)[1]
        return DATA_ROOT / Path(relative_part)

    fallback = PROJECT_ROOT / raw_text
    return fallback if fallback.exists() else None


def _asset_url(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        relative = path.resolve().relative_to(DATA_ROOT.resolve())
    except ValueError:
        return None
    return f"/assets/data/{relative.as_posix()}"


def _normalize_asset_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for column in PATH_COLUMNS:
        resolved = _resolve_project_asset_path(normalized.get(column))
        normalized[column] = _asset_url(resolved)
    normalized["timestamp"] = _iso_timestamp(normalized.get("timestamp"))
    normalized["mask_ratio"] = _safe_float(normalized.get("mask_ratio"), 4)
    normalized["deformation_energy"] = _safe_float(normalized.get("deformation_energy"), 4)
    normalized["coherence_mean"] = _safe_float(normalized.get("coherence_mean"), 4)
    return normalized


def _fleet_predictions() -> pd.DataFrame:
    ensure_core_artifacts()
    predictions = _read_csv(BRIDGE_PREDICTIONS_PATH)
    if predictions.empty:
        return predictions
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce")
    return predictions.dropna(subset=["timestamp"])


def _bridge_registry() -> pd.DataFrame:
    ensure_core_artifacts()
    return _read_csv(BRIDGE_REGISTRY_PATH)


def _model_metrics() -> dict[str, Any]:
    ensure_core_artifacts()
    return _read_json(BRIDGE_MODEL_METRICS_PATH)


def _bridge_directory(bridge_id: str) -> Path:
    bridge_dir = BRIDGES_DIR / bridge_id
    if not bridge_dir.exists():
        raise FileNotFoundError(f"Unknown bridge: {bridge_id}")
    return bridge_dir


def _bridge_label(bridge_id: str) -> str:
    registry = _bridge_registry()
    match = registry[registry["bridge_id"] == bridge_id]
    if match.empty:
        return bridge_id
    return str(match.iloc[0]["bridge_name"])


def _bridge_predictions(bridge_id: str) -> pd.DataFrame:
    bridge_dir = _bridge_directory(bridge_id)
    path = bridge_dir / "predictions.csv"
    predictions = _read_csv(path)
    if predictions.empty:
        predictions, _ = run_bridge_inference(bridge_id, return_trace=True)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce")
    return predictions.dropna(subset=["timestamp"]).reset_index(drop=True)


def _timeseries_gnss(bridge_dir: Path) -> list[dict[str, Any]]:
    gnss = _read_csv(bridge_dir / "gnss_raw.csv")
    if gnss.empty:
        return []
    gnss["timestamp"] = pd.to_datetime(gnss["timestamp"], errors="coerce")
    gnss = gnss.dropna(subset=["timestamp"]).reset_index(drop=True)
    baseline = gnss.loc[0, ["x", "y", "z"]].astype(float)
    offsets = gnss[["x", "y", "z"]].astype(float).subtract(baseline)
    gnss["total_mm"] = (offsets.pow(2).sum(axis=1).pow(0.5) * 1000.0)
    return [
        {
            "timestamp": row["timestamp"].isoformat(),
            "x": _safe_float(row["x"], 5),
            "y": _safe_float(row["y"], 5),
            "z": _safe_float(row["z"], 5),
            "total_mm": _safe_float(row["total_mm"], 3),
        }
        for _, row in gnss.iterrows()
    ]


def _timeseries_insar(bridge_dir: Path) -> list[dict[str, Any]]:
    insar = _read_csv(bridge_dir / "insar_timeseries.csv")
    if insar.empty:
        return []
    insar["timestamp"] = pd.to_datetime(insar["timestamp"], errors="coerce")
    insar = insar.dropna(subset=["timestamp"])
    return [
        {
            "timestamp": row["timestamp"].isoformat(),
            "los_displacement": _safe_float(row.get("los_displacement"), 3),
        }
        for _, row in insar.iterrows()
    ]


def _timeseries_sensors(bridge_dir: Path) -> list[dict[str, Any]]:
    sensors = _read_csv(bridge_dir / "sensor_data.csv")
    if sensors.empty:
        return []
    sensors["timestamp"] = pd.to_datetime(sensors["timestamp"], errors="coerce")
    sensors = sensors.dropna(subset=["timestamp"])
    focus_columns = [
        "Strain_microstrain",
        "Deflection_mm",
        "Vibration_ms2",
        "Tilt_deg",
        "Temperature_C",
        "Humidity_percent",
        "Probability_of_Failure_PoF",
        "Structural_Health_Index_SHI",
    ]
    available = [column for column in focus_columns if column in sensors.columns]
    rows: list[dict[str, Any]] = []
    for _, row in sensors.iterrows():
        item: dict[str, Any] = {"timestamp": row["timestamp"].isoformat()}
        for column in available:
            item[column] = _safe_float(row.get(column), 3)
        rows.append(item)
    return rows


def _bridge_insar_frames(bridge_dir: Path) -> list[dict[str, Any]]:
    frames = _read_csv(bridge_dir / "insar_mask_metadata.csv")
    if frames.empty:
        return []
    return [_normalize_asset_row(row) for row in frames.to_dict(orient="records")]


def _bridge_xai(bridge_dir: Path) -> list[dict[str, Any]]:
    xai = _read_csv(bridge_dir / "xai_top_factors.csv")
    if xai.empty:
        return []
    return [
        {
            "feature": str(row.get("feature", "")),
            "impact": _safe_float(row.get("impact"), 12),
            "baseline_probability": _safe_float(row.get("baseline_probability"), 9),
            "counterfactual_probability": _safe_float(row.get("counterfactual_probability"), 9),
        }
        for _, row in xai.iterrows()
    ]


def _score_status(score: int) -> str:
    if score >= 80:
        return "verified"
    if score >= 60:
        return "review"
    return "weak"


def _check_status(score: int) -> str:
    if score >= 80:
        return "pass"
    if score >= 55:
        return "watch"
    return "fail"


def _bridge_validation(predictions: pd.DataFrame, xai_factors: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    if predictions.empty:
        return {
            "status": "weak",
            "score": 0,
            "confidence_band": "Unavailable",
            "consensus_level": "Unavailable",
            "evidence_summary": ["No prediction frame is available for model output verification."],
            "checks": [],
        }

    anomaly_rows = predictions[predictions["anomaly"] == 1].copy()
    ranked = predictions.sort_values("anomaly_probability", ascending=False).reset_index(drop=True)
    top_row = ranked.iloc[0]
    top_probability = float(top_row.get("anomaly_probability") or 0.0)
    top_slice = ranked.head(min(10, len(ranked)))
    mean_top_probability = float(top_slice["anomaly_probability"].mean()) if "anomaly_probability" in top_slice else 0.0

    precision = float(metrics.get("precision") or 0.0)
    recall = float(metrics.get("recall") or 0.0)
    average_precision = float(metrics.get("average_precision") or 0.0)
    performance_score = int(round(((precision + recall + average_precision) / 3.0) * 100))

    support_columns = {
        "Deflection_mm": "deflection",
        "Displacement_mm": "displacement",
        "Vibration_ms2": "vibration",
        "Probability_of_Failure_PoF": "failure probability",
    }
    agreeing_modalities: list[str] = []
    for column, label in support_columns.items():
        if column not in predictions.columns:
            continue
        series = pd.to_numeric(predictions[column], errors="coerce").dropna()
        if series.empty:
            continue
        threshold = float(series.quantile(0.9))
        candidate = pd.to_numeric(pd.Series([top_row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(candidate) and float(candidate) >= threshold:
            agreeing_modalities.append(label)
    modality_count = len(agreeing_modalities)
    if modality_count >= 4:
        agreement_score = 100
    elif modality_count == 3:
        agreement_score = 86
    elif modality_count == 2:
        agreement_score = 68
    elif modality_count == 1:
        agreement_score = 42
    else:
        agreement_score = 18

    confidence_score = int(round(min(1.0, (top_probability * 0.65) + (mean_top_probability * 0.35)) * 100))

    top_impacts = [abs(float(item.get("impact") or 0.0)) for item in xai_factors[:8]]
    impact_total = sum(top_impacts)
    impact_shares = [impact / impact_total for impact in top_impacts if impact_total > 0 and impact > 0]
    diversified_factors = sum(1 for share in impact_shares if share >= 0.08)
    avg_counterfactual_drop = 0.0
    if xai_factors:
        drops = [
            max(0.0, float(item.get("baseline_probability") or 0.0) - float(item.get("counterfactual_probability") or 0.0))
            for item in xai_factors[:8]
        ]
        avg_counterfactual_drop = sum(drops) / len(drops)
    diversification_component = min(1.0, diversified_factors / 3.0)
    counterfactual_component = min(1.0, avg_counterfactual_drop / 0.0005)
    explainability_score = int(round(min(1.0, diversification_component * 0.75 + counterfactual_component * 0.25) * 100))

    hotspot_consistency_score = 0
    hotspot_detail = "No anomaly hotspot distribution available."
    if not anomaly_rows.empty and "Vibration_Anomaly_Location" in anomaly_rows.columns:
        hotspot_counts = anomaly_rows["Vibration_Anomaly_Location"].fillna("Deck").astype(str).value_counts()
        dominant_share = float(hotspot_counts.iloc[0] / max(1, hotspot_counts.sum()))
        hotspot_consistency_score = int(round(min(1.0, dominant_share / 0.55) * 100))
        hotspot_detail = f"Dominant hotspot {hotspot_counts.index[0]} appears in {(dominant_share * 100):.1f}% of anomalous rows."

    overall_score = int(round(
        performance_score * 0.3
        + confidence_score * 0.25
        + agreement_score * 0.2
        + explainability_score * 0.15
        + hotspot_consistency_score * 0.1
    ))
    status = _score_status(overall_score)

    evidence_summary = [
        f"Model quality gate: precision {precision:.3f}, recall {recall:.3f}, PR-AUC {average_precision:.3f}.",
        f"Peak anomaly probability is {top_probability:.1%} with a top-window mean of {mean_top_probability:.1%}.",
        f"Cross-signal agreement detected across {len(agreeing_modalities)} supporting modalities: {', '.join(agreeing_modalities) if agreeing_modalities else 'none'}.",
        hotspot_detail,
    ]

    return {
        "status": status,
        "score": overall_score,
        "confidence_band": "High" if confidence_score >= 80 else "Moderate" if confidence_score >= 60 else "Low",
        "consensus_level": "Strong" if modality_count >= 3 else "Moderate" if modality_count == 2 else "Weak",
        "evidence_summary": evidence_summary,
        "checks": [
            {
                "name": "Model Quality Gate",
                "status": _check_status(performance_score),
                "score": performance_score,
                "detail": f"Precision {precision:.3f}, recall {recall:.3f}, PR-AUC {average_precision:.3f}.",
            },
            {
                "name": "Confidence Stability",
                "status": _check_status(confidence_score),
                "score": confidence_score,
                "detail": f"Peak anomaly probability {top_probability:.1%}; top-window mean {mean_top_probability:.1%}.",
            },
            {
                "name": "Cross-Modal Agreement",
                "status": _check_status(agreement_score),
                "score": agreement_score,
                "detail": f"Supporting modalities: {', '.join(agreeing_modalities) if agreeing_modalities else 'none'}",
            },
            {
                "name": "Explainability Support",
                "status": _check_status(explainability_score),
                "score": explainability_score,
                "detail": f"{diversified_factors} drivers exceed meaningful contribution share; average counterfactual drop {avg_counterfactual_drop:.6f}.",
            },
            {
                "name": "Hotspot Consistency",
                "status": _check_status(hotspot_consistency_score),
                "score": hotspot_consistency_score,
                "detail": hotspot_detail,
            },
        ],
    }


def _bridge_hotspots(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    anomaly_rows = predictions[predictions["anomaly"] == 1].copy()
    if anomaly_rows.empty:
        return []

    anomaly_rows["zone_key"] = anomaly_rows["Vibration_Anomaly_Location"].fillna("Deck").astype(str)
    grouped = (
        anomaly_rows.groupby("zone_key")
        .agg(
            hit_count=("anomaly", "sum"),
            max_probability=("anomaly_probability", "max"),
            mean_stress=("Simulated_Localized_Stress_Index", "mean"),
        )
        .reset_index()
        .sort_values(["max_probability", "hit_count"], ascending=False)
    )

    hotspots: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        zone_meta = HOTSPOT_ZONE_LOOKUP.get(row["zone_key"], {"label": row["zone_key"], "x": 50, "y": 50})
        hotspots.append(
            {
                "zone": zone_meta["label"],
                "x": zone_meta["x"],
                "y": zone_meta["y"],
                "hit_count": _safe_int(row["hit_count"]),
                "max_probability": _safe_float(row["max_probability"], 4),
                "mean_stress": _safe_float(row["mean_stress"], 4),
            }
        )
    return hotspots


def _stage_trace_for_bridge(bridge_id: str) -> list[dict[str, Any]]:
    _, runtime = run_bridge_inference(bridge_id, return_trace=True)
    stages = runtime.get("stages", [])
    return [
        {
            "stage": stage.get("stage"),
            "detail": stage.get("detail"),
            "duration_ms": _safe_float(stage.get("duration_ms"), 2),
            "meta": {key: value for key, value in stage.items() if key not in {"stage", "detail", "duration_ms"}},
        }
        for stage in stages
    ]


def _bridge_overview_summary(bridge_row: dict[str, Any], predictions: pd.DataFrame) -> dict[str, Any]:
    anomaly_rows = predictions[predictions["anomaly"] == 1]
    latest_row = predictions.sort_values("timestamp").iloc[-1]
    return {
        "bridge_id": bridge_row["bridge_id"],
        "bridge_name": bridge_row["bridge_name"],
        "city": bridge_row["city"],
        "region": bridge_row["region"],
        "lat": _safe_float(bridge_row["lat"], 4),
        "lon": _safe_float(bridge_row["lon"], 4),
        "latest_timestamp": _iso_timestamp(latest_row["timestamp"]),
        "anomaly_count": _safe_int(anomaly_rows["anomaly"].sum()),
        "max_probability": _safe_float(predictions["anomaly_probability"].max(), 4),
        "avg_probability": _safe_float(predictions["anomaly_probability"].mean(), 4),
        "mean_health_index": _safe_float(predictions["Structural_Health_Index_SHI"].mean(), 4),
        "peak_deflection_mm": _safe_float(predictions["Deflection_mm"].max(), 3),
        "peak_displacement_mm": _safe_float(predictions["Displacement_mm"].max(), 3),
        "top_hotspot": (
            anomaly_rows["Vibration_Anomaly_Location"].dropna().astype(str).mode().iloc[0]
            if "Vibration_Anomaly_Location" in anomaly_rows and not anomaly_rows["Vibration_Anomaly_Location"].dropna().empty
            else "Deck"
        ),
    }


def _report_companion_path(markdown_path: Path) -> Path:
    return markdown_path.with_suffix(".dashboard.json")


def _report_title_from_markdown(content: str, fallback: str) -> str:
    return next((line.replace("#", "").strip() for line in content.splitlines() if line.startswith("#")), fallback)


def _parse_priority_actions(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    priorities = artifact.get("triage", {}).get("bridge_priorities", [])
    parsed: list[dict[str, Any]] = []
    for item in priorities:
        parsed.append(
            {
                "bridge_id": item.get("bridge_id"),
                "bridge_name": item.get("bridge_name"),
                "risk_level": item.get("risk_level"),
                "urgency_score": _safe_int(item.get("urgency_score")),
                "recommendation": item.get("recommendation"),
                "rationale": item.get("rationale"),
                "dominant_hotspot": item.get("dominant_hotspot"),
                "primary_driver": item.get("primary_driver"),
                "modalities_agreeing": item.get("modalities_agreeing") or [],
                "anomaly_count": _safe_int(item.get("anomaly_count")),
                "max_probability": _safe_float(item.get("max_probability"), 4),
            }
        )
    return parsed


def _build_dashboard_payload(report_name: str, artifact: dict[str, Any], markdown_content: str) -> dict[str, Any]:
    analyst = artifact.get("analyst", {})
    triage = artifact.get("triage", {})
    visual_plan = artifact.get("visual_plan", {})
    report = artifact.get("report", {})
    priorities = _parse_priority_actions(artifact)
    focus_bridges = artifact.get("metadata", {}).get("focus_bridges") or []
    overview_bridges = artifact.get("metadata", {}).get("overview_bridges") or []

    risk_distribution = []
    for level in ("High", "Medium", "Low"):
        count = sum(1 for item in priorities if str(item.get("risk_level", "")).lower() == level.lower())
        risk_distribution.append({"label": level, "count": count})

    recommendation_distribution: dict[str, int] = {}
    for item in priorities:
        recommendation = str(item.get("recommendation") or "Monitor")
        recommendation_distribution[recommendation] = recommendation_distribution.get(recommendation, 0) + 1

    driver_breakdown: list[dict[str, Any]] = []
    for item in priorities[:6]:
        driver_breakdown.append(
            {
                "bridge_name": item.get("bridge_name"),
                "driver": item.get("primary_driver") or "Unknown",
                "urgency_score": _safe_int(item.get("urgency_score")),
                "risk_level": item.get("risk_level"),
            }
        )

    hotspot_mix: dict[str, int] = {}
    for item in priorities:
        hotspot = str(item.get("dominant_hotspot") or "Deck")
        hotspot_mix[hotspot] = hotspot_mix.get(hotspot, 0) + 1

    bridge_risk_series = []
    for bridge in overview_bridges:
        bridge_risk_series.append(
            {
                "bridge_name": bridge.get("bridge_name"),
                "max_probability": _safe_float(bridge.get("max_probability"), 4),
                "anomaly_count": _safe_int(bridge.get("anomaly_count")),
                "top_hotspot": bridge.get("top_hotspot"),
            }
        )

    return {
        "hero": {
            "title": visual_plan.get("hero_title") or report.get("report_title") or report_name,
            "message": visual_plan.get("hero_message") or analyst.get("fleet_status") or "CrewAI dashboard run completed.",
            "summary_points": analyst.get("summary_points") or [],
            "watchlist": analyst.get("watchlist") or [],
        },
        "kpis": [
            {"label": "Focus Bridges", "value": str(len(focus_bridges)), "detail": "Bridges reviewed deeply by the crew."},
            {"label": "Priority Actions", "value": str(len(priorities)), "detail": "Bridges returned with explicit intervention guidance."},
            {"label": "High Risk Flags", "value": str(next((item["count"] for item in risk_distribution if item["label"] == "High"), 0)), "detail": "Crew-ranked high risk bridges."},
            {"label": "Admin Prompts", "value": str(len(visual_plan.get("operator_prompts") or [])), "detail": "Workflow prompts generated for operators."},
        ],
        "priority_actions": priorities,
        "risk_distribution": risk_distribution,
        "recommendation_distribution": [{"label": key, "count": value} for key, value in recommendation_distribution.items()],
        "driver_breakdown": driver_breakdown,
        "bridge_risk_series": bridge_risk_series,
        "hotspot_mix": [{"label": key, "count": value} for key, value in hotspot_mix.items()],
        "agent_panels": [
            {
                "agent": "Fleet Intelligence Analyst",
                "headline": analyst.get("headline") or "Fleet intelligence synthesis",
                "body": analyst.get("operational_note") or analyst.get("model_commentary") or "",
                "bullets": analyst.get("summary_points") or [],
            },
            {
                "agent": "Structural Triage Engineer",
                "headline": triage.get("escalation_headline") or "Bridge triage queue",
                "body": "Risk ranking and intervention guidance for monitored bridges.",
                "bullets": triage.get("consensus_notes") or [],
            },
            {
                "agent": "Analytics Dashboard Strategist",
                "headline": visual_plan.get("hero_title") or "Dashboard visual brief",
                "body": visual_plan.get("hero_message") or "",
                "bullets": visual_plan.get("panel_callouts") or [],
            },
            {
                "agent": "Executive Reporting Lead",
                "headline": report.get("report_title") or "Archival engineering brief",
                "body": report.get("executive_brief") or "",
                "bullets": report.get("admin_recommendations") or [],
            },
        ],
        "chart_annotations": visual_plan.get("chart_annotations") or [],
        "operator_prompts": visual_plan.get("operator_prompts") or [],
        "maintenance_queue": triage.get("maintenance_queue") or [],
        "markdown": markdown_content,
    }


def list_reports() -> list[dict[str, Any]]:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted(REPORTS_ROOT.glob("*.md"), reverse=True):
        if path.name == "README.md":
            continue
        content = path.read_text(encoding="utf-8")
        companion = _report_companion_path(path)
        artifact = _read_json(companion) if companion.exists() else {}
        reports.append(
            {
                "name": path.name,
                "title": artifact.get("report", {}).get("report_title") or _report_title_from_markdown(content, path.stem),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "size_bytes": path.stat().st_size,
                "kind": "crew_dashboard" if artifact else "markdown_report",
                "provider": artifact.get("metadata", {}).get("provider"),
                "model": artifact.get("metadata", {}).get("model"),
            }
        )
    return reports


def get_report_content(name: str) -> dict[str, Any]:
    path = REPORTS_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Unknown report: {name}")
    content = path.read_text(encoding="utf-8")
    companion = _report_companion_path(path)
    artifact = _read_json(companion) if companion.exists() else {}
    return {
        "name": path.name,
        "content": content,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        "kind": "crew_dashboard" if artifact else "markdown_report",
        "metadata": artifact.get("metadata", {}),
        "dashboard": _build_dashboard_payload(path.name, artifact, content) if artifact else None,
    }


def generate_report() -> dict[str, Any]:
    overview = get_fleet_overview()
    lines = [
        "# Executive Summary",
        f"The monitoring fleet currently covers {overview['fleet_metrics']['total_bridges']} bridges with {overview['fleet_metrics']['bridges_with_alerts']} bridges carrying active anomaly flags.",
        "",
        "## Fleet Snapshot",
        f"- Peak anomaly probability across the fleet: {overview['fleet_metrics']['peak_probability']:.4f}",
        f"- Mean anomaly probability: {overview['fleet_metrics']['average_probability']:.4f}",
        f"- Highest risk bridge: {overview['fleet_metrics']['highest_risk_bridge']}",
        "",
        "## Bridge Prioritization",
    ]

    for bridge in overview["bridges"][:6]:
        lines.extend(
            [
                f"### {bridge['bridge_name']} ({bridge['bridge_id']})",
                f"- Location: {bridge['city']}, {bridge['region']}",
                f"- Active anomaly hits: {bridge['anomaly_count']}",
                f"- Max anomaly probability: {bridge['max_probability']:.4f}",
                f"- Peak deflection: {bridge['peak_deflection_mm']:.3f} mm",
                f"- Dominant hotspot: {bridge['top_hotspot']}",
                "",
            ]
        )

    metrics = overview["model_metrics"]
    lines.extend(
        [
            "## Model Health",
            f"- Precision: {metrics.get('precision', 0):.3f}",
            f"- Recall: {metrics.get('recall', 0):.3f}",
            f"- F1 score: {metrics.get('f1', 0):.3f}",
            f"- PR-AUC: {metrics.get('average_precision', 0):.3f}",
            "",
            "## Recommendation",
            "Focus inspection scheduling on the top-ranked bridges first, and use the bridge detail views to review telemetry trends, InSAR masks, and explainability drivers before dispatch.",
        ]
    )

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    report_name = f"shm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_web.md"
    report_path = REPORTS_ROOT / report_name
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return get_report_content(report_name)


def start_crew_report_operation(
    provider: str = "gemini",
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    provider = os.environ.get("CREWAI_PROVIDER", provider)
    model = os.environ.get("GEMINI_MODEL") or model or "gemini/gemini-2.5-pro"
    operation = _create_operation("crew_report", "CrewAI Analytics")
    _set_operation(operation["id"], message="Preparing CrewAI analytics crew...", progress=2)

    def worker() -> None:
        _append_operation_step(operation["id"], _progress_step("Queued", "CrewAI analytics request accepted.", "completed"))
        _set_operation(operation["id"], message="Collecting fleet context for the CrewAI manager...", progress=8)

        overview = get_fleet_overview()
        pipeline_summary = {
            "overview": overview,
            "requested_at": datetime.utcnow().isoformat(),
        }
        _append_operation_step(
            operation["id"],
            _progress_step(
                "Context Assembly",
                "Fleet overview, model metrics, and report registry prepared for the crew manager.",
                "completed",
                {"bridges": len(overview.get("bridges", []))},
            ),
        )

        stage_progress = {
            "Crew Boot": 16,
            "Crew Dispatch": 26,
            "Crew Tool": 42,
            "Crew Synthesis": 86,
            "Crew Archive": 96,
        }

        def on_progress(stage_payload: dict[str, Any]) -> None:
            stage_name = str(stage_payload.get("stage") or "CrewAI")
            detail = str(stage_payload.get("detail") or "CrewAI progress update.")
            _append_operation_step(
                operation["id"],
                _progress_step(stage_name, detail, "completed", stage_payload.get("meta") or {}),
            )
            current_progress = get_operation(operation["id"])["progress"]
            next_progress = stage_progress.get(stage_name, min(94, int(current_progress) + 6))
            _set_operation(operation["id"], message=detail, progress=next_progress)

        from src.agents.crew import kickoff_shm_crew_dashboard

        result = kickoff_shm_crew_dashboard(
            pipeline_summary,
            provider=provider,
            api_key=api_key,
            model=model,
            progress_callback=on_progress,
        )

        REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
        markdown_path = REPORTS_ROOT / f"{result.artifact_name}.md"
        companion_path = _report_companion_path(markdown_path)
        companion_path.write_text(json.dumps(result.artifact, indent=2), encoding="utf-8")
        report = get_report_content(markdown_path.name)

        _set_operation(
            operation["id"],
            status="completed",
            message="CrewAI analytics dashboard generated successfully.",
            progress=100,
            result={"report": report},
        )

    _run_operation_in_thread(operation["id"], worker)
    return operation


def get_fleet_overview() -> dict[str, Any]:
    registry = _bridge_registry()
    predictions = _fleet_predictions()
    metrics = _model_metrics()

    grouped = (
        predictions.groupby("bridge_id")
        .agg(
            anomaly_count=("anomaly", "sum"),
            avg_probability=("anomaly_probability", "mean"),
            max_probability=("anomaly_probability", "max"),
            mean_health_index=("Structural_Health_Index_SHI", "mean"),
            peak_deflection_mm=("Deflection_mm", "max"),
            peak_displacement_mm=("Displacement_mm", "max"),
        )
        .reset_index()
    )
    bridges = registry.merge(grouped, on="bridge_id", how="left").fillna(0.0)

    hotspot_lookup = (
        predictions[predictions["anomaly"] == 1]
        .groupby("bridge_id")["Vibration_Anomaly_Location"]
        .agg(lambda values: values.dropna().astype(str).mode().iloc[0] if not values.dropna().empty else "Deck")
        .to_dict()
    )

    bridge_cards = []
    for _, row in bridges.iterrows():
        bridge_cards.append(
            {
                "bridge_id": row["bridge_id"],
                "bridge_name": row["bridge_name"],
                "city": row["city"],
                "region": row["region"],
                "lat": _safe_float(row["lat"], 4),
                "lon": _safe_float(row["lon"], 4),
                "anomaly_count": _safe_int(row["anomaly_count"]),
                "avg_probability": _safe_float(row["avg_probability"], 4),
                "max_probability": _safe_float(row["max_probability"], 4),
                "mean_health_index": _safe_float(row["mean_health_index"], 4),
                "peak_deflection_mm": _safe_float(row["peak_deflection_mm"], 3),
                "peak_displacement_mm": _safe_float(row["peak_displacement_mm"], 3),
                "top_hotspot": hotspot_lookup.get(row["bridge_id"], "Deck"),
            }
        )

    bridge_cards.sort(key=lambda item: (item["max_probability"] or 0.0, item["anomaly_count"]), reverse=True)
    fleet_metrics = {
        "total_bridges": len(bridge_cards),
        "bridges_with_alerts": sum(1 for bridge in bridge_cards if bridge["anomaly_count"] > 0),
        "peak_probability": max((bridge["max_probability"] or 0.0) for bridge in bridge_cards) if bridge_cards else 0.0,
        "average_probability": (
            round(sum((bridge["avg_probability"] or 0.0) for bridge in bridge_cards) / len(bridge_cards), 4)
            if bridge_cards
            else 0.0
        ),
        "highest_risk_bridge": bridge_cards[0]["bridge_name"] if bridge_cards else "Unavailable",
    }

    return {
        "fleet_metrics": fleet_metrics,
        "bridges": bridge_cards,
        "model_metrics": metrics,
        "reports": list_reports(),
    }


def get_bridge_detail(bridge_id: str) -> dict[str, Any]:
    ensure_core_artifacts()
    bridge_dir = _bridge_directory(bridge_id)
    registry = _bridge_registry()
    bridge_row = registry[registry["bridge_id"] == bridge_id]
    if bridge_row.empty:
        raise FileNotFoundError(f"Unknown bridge: {bridge_id}")
    bridge_info = bridge_row.iloc[0].to_dict()

    predictions = _bridge_predictions(bridge_id)
    xai = _bridge_xai(bridge_dir)
    insar_frames = _bridge_insar_frames(bridge_dir)
    telemetry = {
        "gnss": _timeseries_gnss(bridge_dir),
        "insar": _timeseries_insar(bridge_dir),
        "sensors": _timeseries_sensors(bridge_dir),
    }
    runtime_trace = _stage_trace_for_bridge(bridge_id)
    metrics = _model_metrics()

    top_anomalies = predictions.sort_values("anomaly_probability", ascending=False).head(50)
    anomaly_rows = [
        {
            "timestamp": _iso_timestamp(row["timestamp"]),
            "anomaly_probability": _safe_float(row.get("anomaly_probability"), 4),
            "anomaly": _safe_int(row.get("anomaly")),
            "hotspot": str(row.get("Vibration_Anomaly_Location") or "Deck"),
            "strain_hotspot": _safe_float(row.get("Localized_Strain_Hotspot"), 3),
            "deflection_mm": _safe_float(row.get("Deflection_mm"), 3),
            "displacement_mm": _safe_float(row.get("Displacement_mm"), 3),
            "vibration_ms2": _safe_float(row.get("Vibration_ms2"), 3),
            "failure_probability": _safe_float(row.get("Probability_of_Failure_PoF"), 4),
            "health_index": _safe_float(row.get("Structural_Health_Index_SHI"), 4),
        }
        for _, row in top_anomalies.iterrows()
    ]

    summary = _bridge_overview_summary(bridge_info, predictions)
    return {
        "bridge": summary,
        "telemetry": telemetry,
        "insar_frames": insar_frames,
        "xai_factors": xai,
        "validation": _bridge_validation(predictions, xai, metrics),
        "hotspots": _bridge_hotspots(predictions),
        "anomalies": anomaly_rows,
        "runtime_trace": runtime_trace,
        "report_count": len(list_reports()),
    }


def rerun_bridge_analysis(bridge_id: str) -> dict[str, Any]:
    run_bridge_inference(bridge_id)
    return get_bridge_detail(bridge_id)


def rerun_full_pipeline(generate_synthetic_data: bool = False, run_agents: bool = False) -> dict[str, Any]:
    summary = {"synthetic": run_pipeline(generate_synthetic_data=generate_synthetic_data)}
    summary["kaggle"] = run_kaggle_bridge_pipeline()
    if run_agents:
        try:
            from src.agents.crew import kickoff_shm_crew

            summary["agents"] = {"status": "started"}
            kickoff_shm_crew(summary, provider="openai")
        except Exception as exc:
            summary["agents"] = {"status": "failed", "message": str(exc)}
    return {"summary": summary, "overview": get_fleet_overview()}


def start_bridge_refresh_operation(bridge_id: str) -> dict[str, Any]:
    _bridge_directory(bridge_id)
    bridge_name = _bridge_label(bridge_id)
    operation = _create_operation("bridge_refresh", bridge_name)
    _set_operation(operation["id"], message=f"Refreshing {bridge_name}...", progress=2)

    def worker() -> None:
        _set_operation(operation["id"], message=f"Refreshing {bridge_name}...", progress=5)
        _append_operation_step(operation["id"], _progress_step("Queued", f"{bridge_name} refresh request accepted.", "completed"))

        stage_progress = {
            "Data Intake": 18,
            "Modal Sync": 34,
            "Feature Synthesis": 52,
            "Ensemble Scoring": 70,
            "Explainability": 86,
            "Spatial Projection": 96,
        }

        def on_progress(stage_payload: dict[str, Any]) -> None:
            stage_name = str(stage_payload.get("stage") or "Bridge Analysis")
            _append_operation_step(
                operation["id"],
                _progress_step(
                    stage_name,
                    str(stage_payload.get("detail") or ""),
                    "completed",
                    stage_payload.get("meta") or {},
                ),
            )
            _set_operation(
                operation["id"],
                message=str(stage_payload.get("detail") or f"Completed {stage_name}."),
                progress=stage_progress.get(stage_name, 90),
            )

        run_bridge_inference(bridge_id, progress_callback=on_progress)
        detail = get_bridge_detail(bridge_id)
        _set_operation(
            operation["id"],
            status="completed",
            message=f"{bridge_name} refresh completed.",
            progress=100,
            result={"detail": detail},
        )

    _run_operation_in_thread(operation["id"], worker)
    return operation


def start_pipeline_operation(generate_synthetic_data: bool = False) -> dict[str, Any]:
    operation = _create_operation("full_pipeline")
    _set_operation(operation["id"], message="Preparing full structural pipeline...", progress=2)

    def worker() -> None:
        _append_operation_step(operation["id"], _progress_step("Queued", "Pipeline run request accepted.", "completed"))
        if generate_synthetic_data:
            _set_operation(operation["id"], message="Generating synthetic telemetry sources...", progress=10)
            _append_operation_step(
                operation["id"],
                _progress_step("Synthetic Data", "Generating GNSS, InSAR, and sensor telemetry inputs.", "running"),
            )
        else:
            _set_operation(operation["id"], message="Preparing existing telemetry inputs...", progress=10)
            _append_operation_step(
                operation["id"],
                _progress_step("Input Reuse", "Using existing telemetry files and skipping synthetic generation.", "completed"),
            )

        synthetic_summary = run_pipeline(generate_synthetic_data=generate_synthetic_data)
        _append_operation_step(
            operation["id"],
            _progress_step(
                "Synthetic Pipeline",
                "GNSS, InSAR, sensors, fusion, anomaly detection, and image processing completed.",
                "completed",
                synthetic_summary,
            ),
        )
        _set_operation(operation["id"], message="Training bridge ensemble and refreshing fleet artifacts...", progress=58)

        kaggle_summary = run_kaggle_bridge_pipeline()
        _append_operation_step(
            operation["id"],
            _progress_step(
                "Bridge Ensemble",
                "Bridge anomaly model training, predictions, and bridge inference export completed.",
                "completed",
                kaggle_summary,
            ),
        )
        _set_operation(operation["id"], message="Refreshing overview artifacts and reports index...", progress=88)

        overview = get_fleet_overview()
        _append_operation_step(
            operation["id"],
            _progress_step(
                "Fleet Overview",
                "Overview, metrics, and report metadata were refreshed for the admin workspace.",
                "completed",
                {"bridges": len(overview["bridges"])},
            ),
        )
        _set_operation(
            operation["id"],
            status="completed",
            message="Full structural pipeline completed.",
            progress=100,
            result={"summary": {"synthetic": synthetic_summary, "kaggle": kaggle_summary}, "overview": overview},
        )

    _run_operation_in_thread(operation["id"], worker)
    return operation
