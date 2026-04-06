from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from src.utils.config import (
    BRIDGES_DIR,
    BRIDGE_MODEL_METRICS_PATH,
    BRIDGE_PREDICTIONS_PATH,
    BRIDGE_REGISTRY_PATH,
)


class EmptyInput(BaseModel):
    tool_input: str = Field(default="read", description="Pass the string 'read'.")


class BridgeInput(BaseModel):
    bridge_id: str = Field(..., description="The bridge_id to inspect.")


class WriteReportInput(BaseModel):
    markdown_content: str = Field(..., description="The full markdown report content.")
    file_stem: str = Field(default="crew_dashboard", description="The file stem to use when writing the report.")


class ProgressAwareTool(BaseTool):
    _progress_callback: Callable[[dict[str, Any]], None] | None = PrivateAttr(default=None)

    def __init__(self, progress_callback: Callable[[dict[str, Any]], None] | None = None, **data: Any):
        super().__init__(**data)
        self._progress_callback = progress_callback

    def _notify(self, stage: str, detail: str, **meta: Any) -> None:
        if self._progress_callback:
            self._progress_callback(
                {
                    "stage": stage,
                    "detail": detail,
                    "meta": meta,
                }
            )


def _safe_float(value: Any, digits: int = 4) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return round(result, digits)


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_predictions() -> pd.DataFrame:
    if not BRIDGE_PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(BRIDGE_PREDICTIONS_PATH)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


def _load_registry() -> pd.DataFrame:
    if not BRIDGE_REGISTRY_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(BRIDGE_REGISTRY_PATH)


class ReadFleetOverviewTool(ProgressAwareTool):
    name: str = "read_fleet_overview_tool"
    description: str = "Reads the fleet bridge registry and prediction outputs, then returns a compact JSON overview for the monitoring portfolio."
    args_schema: type[BaseModel] = EmptyInput

    def _run(self, tool_input: str = "read") -> str:
        self._notify("Crew Tool", "Reading fleet overview and bridge portfolio.", tool=self.name)
        registry = _load_registry()
        predictions = _load_predictions()
        if registry.empty or predictions.empty:
            return json.dumps({"error": "Fleet overview inputs are unavailable."}, indent=2)

        grouped = (
            predictions.groupby("bridge_id")
            .agg(
                anomaly_count=("anomaly", "sum"),
                max_probability=("anomaly_probability", "max"),
                avg_probability=("anomaly_probability", "mean"),
                mean_health_index=("Structural_Health_Index_SHI", "mean"),
                top_hotspot=("Vibration_Anomaly_Location", lambda values: values.dropna().astype(str).mode().iloc[0] if not values.dropna().empty else "Deck"),
            )
            .reset_index()
        )
        merged = registry.merge(grouped, on="bridge_id", how="left").fillna(0)
        merged = merged.sort_values(["max_probability", "anomaly_count"], ascending=False)

        payload = {
            "generated_at": datetime.utcnow().isoformat(),
            "fleet_metrics": {
                "bridge_count": int(len(merged)),
                "bridges_with_alerts": int((merged["anomaly_count"] > 0).sum()),
                "peak_probability": _safe_float(merged["max_probability"].max()),
                "average_probability": _safe_float(merged["avg_probability"].mean()),
            },
            "bridges": [
                {
                    "bridge_id": row["bridge_id"],
                    "bridge_name": row["bridge_name"],
                    "city": row["city"],
                    "region": row["region"],
                    "anomaly_count": _safe_int(row["anomaly_count"]),
                    "max_probability": _safe_float(row["max_probability"]),
                    "avg_probability": _safe_float(row["avg_probability"]),
                    "mean_health_index": _safe_float(row["mean_health_index"]),
                    "top_hotspot": str(row["top_hotspot"] or "Deck"),
                }
                for _, row in merged.iterrows()
            ],
        }
        self._notify("Crew Tool", "Fleet overview prepared for the crew.", tool=self.name, bridges=len(payload["bridges"]))
        return json.dumps(payload, indent=2)


class ReadBridgePredictionsTool(ProgressAwareTool):
    name: str = "read_bridge_predictions_tool"
    description: str = "Reads structural anomaly predictions across all bridges and returns a ranked JSON list of bridges with the strongest alert activity."
    args_schema: type[BaseModel] = EmptyInput

    def _run(self, tool_input: str = "read") -> str:
        self._notify("Crew Tool", "Ranking bridge anomaly predictions.", tool=self.name)
        predictions = _load_predictions()
        if predictions.empty:
            return json.dumps({"bridges": [], "message": "No anomaly prediction data available."}, indent=2)

        registry = _load_registry()
        anomaly_rows = predictions[predictions.get("anomaly", 0) == 1].copy()
        if anomaly_rows.empty:
            return json.dumps({"bridges": [], "message": "No anomalous bridges detected in the current dataset."}, indent=2)

        grouped = (
            anomaly_rows.groupby("bridge_id")
            .agg(
                anomaly_count=("anomaly", "sum"),
                max_probability=("anomaly_probability", "max"),
                mean_probability=("anomaly_probability", "mean"),
                peak_deflection_mm=("Deflection_mm", "max"),
                peak_displacement_mm=("Displacement_mm", "max"),
                peak_vibration_ms2=("Vibration_ms2", "max"),
                peak_failure_probability=("Probability_of_Failure_PoF", "max"),
                dominant_hotspot=("Vibration_Anomaly_Location", lambda values: values.dropna().astype(str).mode().iloc[0] if not values.dropna().empty else "Deck"),
            )
            .reset_index()
        )
        merged = registry.merge(grouped, on="bridge_id", how="right").sort_values(
            ["max_probability", "anomaly_count"], ascending=False
        )
        payload = {
            "bridges": [
                {
                    "bridge_id": row["bridge_id"],
                    "bridge_name": row["bridge_name"],
                    "city": row["city"],
                    "region": row["region"],
                    "anomaly_count": _safe_int(row["anomaly_count"]),
                    "max_probability": _safe_float(row["max_probability"]),
                    "mean_probability": _safe_float(row["mean_probability"]),
                    "peak_deflection_mm": _safe_float(row["peak_deflection_mm"], 3),
                    "peak_displacement_mm": _safe_float(row["peak_displacement_mm"], 3),
                    "peak_vibration_ms2": _safe_float(row["peak_vibration_ms2"], 3),
                    "peak_failure_probability": _safe_float(row["peak_failure_probability"]),
                    "dominant_hotspot": str(row["dominant_hotspot"] or "Deck"),
                }
                for _, row in merged.iterrows()
            ]
        }
        self._notify("Crew Tool", "Bridge anomaly ranking prepared.", tool=self.name, bridges=len(payload["bridges"]))
        return json.dumps(payload, indent=2)


class ReadBridgeTelemetryTool(ProgressAwareTool):
    name: str = "read_bridge_telemetry_tool"
    description: str = "Reads GNSS, InSAR, and sensor telemetry summaries for a specific bridge and returns a compact JSON summary."
    args_schema: type[BaseModel] = BridgeInput

    def _run(self, bridge_id: str) -> str:
        self._notify("Crew Tool", f"Reading telemetry bundle for {bridge_id}.", tool=self.name, bridge_id=bridge_id)
        bridge_dir = BRIDGES_DIR / bridge_id
        if not bridge_dir.exists():
            return json.dumps({"error": f"Unknown bridge {bridge_id}."}, indent=2)

        def read_csv(name: str) -> pd.DataFrame:
            path = bridge_dir / name
            return pd.read_csv(path) if path.exists() else pd.DataFrame()

        gnss = read_csv("gnss_raw.csv")
        insar = read_csv("insar_timeseries.csv")
        sensors = read_csv("sensor_data.csv")
        preds = read_csv("predictions.csv")

        payload = {
            "bridge_id": bridge_id,
            "gnss": {
                "rows": int(len(gnss)),
                "max_vertical": _safe_float(gnss["z"].astype(float).max(), 5) if "z" in gnss else None,
                "min_vertical": _safe_float(gnss["z"].astype(float).min(), 5) if "z" in gnss else None,
            },
            "insar": {
                "rows": int(len(insar)),
                "peak_los_displacement": _safe_float(insar["los_displacement"].astype(float).max(), 3) if "los_displacement" in insar else None,
            },
            "sensors": {
                "rows": int(len(sensors)),
                "peak_deflection_mm": _safe_float(sensors["Deflection_mm"].astype(float).max(), 3) if "Deflection_mm" in sensors else None,
                "peak_vibration_ms2": _safe_float(sensors["Vibration_ms2"].astype(float).max(), 3) if "Vibration_ms2" in sensors else None,
                "peak_pof": _safe_float(sensors["Probability_of_Failure_PoF"].astype(float).max(), 4) if "Probability_of_Failure_PoF" in sensors else None,
            },
            "predictions": {
                "anomaly_rows": _safe_int(preds["anomaly"].sum()) if "anomaly" in preds else 0,
                "max_probability": _safe_float(preds["anomaly_probability"].max(), 4) if "anomaly_probability" in preds else None,
            },
        }
        self._notify("Crew Tool", f"Telemetry summary prepared for {bridge_id}.", tool=self.name, bridge_id=bridge_id)
        return json.dumps(payload, indent=2)


class ReadBridgeXAITool(ProgressAwareTool):
    name: str = "read_bridge_xai_tool"
    description: str = "Reads the top explainability drivers for a specific bridge and returns them as JSON."
    args_schema: type[BaseModel] = BridgeInput

    def _run(self, bridge_id: str) -> str:
        self._notify("Crew Tool", f"Reading explainability drivers for {bridge_id}.", tool=self.name, bridge_id=bridge_id)
        xai_path = BRIDGES_DIR / bridge_id / "xai_top_factors.csv"
        if not xai_path.exists():
            return json.dumps({"bridge_id": bridge_id, "drivers": []}, indent=2)
        df = pd.read_csv(xai_path)
        payload = {
            "bridge_id": bridge_id,
            "drivers": [
                {
                    "feature": str(row.get("feature", "")),
                    "impact": _safe_float(row.get("impact"), 6),
                    "baseline_probability": _safe_float(row.get("baseline_probability"), 6),
                    "counterfactual_probability": _safe_float(row.get("counterfactual_probability"), 6),
                }
                for _, row in df.head(8).iterrows()
            ],
        }
        self._notify("Crew Tool", f"Explainability drivers prepared for {bridge_id}.", tool=self.name, bridge_id=bridge_id)
        return json.dumps(payload, indent=2)


class ReadModelMetricsTool(ProgressAwareTool):
    name: str = "read_model_metrics_tool"
    description: str = "Reads the fleet anomaly model metrics and returns them as JSON."
    args_schema: type[BaseModel] = EmptyInput

    def _run(self, tool_input: str = "read") -> str:
        self._notify("Crew Tool", "Reading anomaly model performance metrics.", tool=self.name)
        if not BRIDGE_MODEL_METRICS_PATH.exists():
            return json.dumps({"error": "Model metrics file not found."}, indent=2)
        metrics = json.loads(BRIDGE_MODEL_METRICS_PATH.read_text(encoding="utf-8"))
        payload = {
            "precision": _safe_float(metrics.get("precision")),
            "recall": _safe_float(metrics.get("recall")),
            "f1": _safe_float(metrics.get("f1")),
            "average_precision": _safe_float(metrics.get("average_precision")),
            "roc_auc": _safe_float(metrics.get("roc_auc")),
            "feature_count": _safe_int(metrics.get("feature_count")),
        }
        self._notify("Crew Tool", "Model metrics prepared for the crew.", tool=self.name)
        return json.dumps(payload, indent=2)


class WriteReportTool(ProgressAwareTool):
    name: str = "write_report_tool"
    description: str = "Writes the final CrewAI markdown briefing into the reports directory and returns the written file path."
    args_schema: type[BaseModel] = WriteReportInput

    def _run(self, markdown_content: str, file_stem: str = "crew_dashboard") -> str:
        self._notify("Crew Tool", "Writing CrewAI markdown briefing to disk.", tool=self.name)
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{file_stem}.md"
        report_path.write_text(markdown_content, encoding="utf-8")
        self._notify("Crew Tool", "CrewAI markdown briefing written successfully.", tool=self.name, path=str(report_path))
        return json.dumps({"path": str(report_path)}, indent=2)
