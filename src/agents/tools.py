import json
from pathlib import Path
import pandas as pd
from crewai.tools import BaseTool

from src.utils.config import (
    GNSS_ANALYSIS_PATH,
    BRIDGE_PREDICTIONS_PATH,
    BRIDGE_MODEL_METRICS_PATH,
    BRIDGES_DIR,
)

from pydantic import BaseModel, Field

class EmptyInput(BaseModel):
    tool_input: str = Field(default="read", description="A required dummy argument. You MUST pass the string 'read'.")

class XAIInput(BaseModel):
    bridge_id: str = Field(..., description="The ID of the bridge to read XAI factors for.")

class WriteReportInput(BaseModel):
    markdown_content: str = Field(..., description="The full markdown content of the engineering report.")

class ReadGNSSAnalysisTool(BaseTool):
    name: str = "read_gnss_analysis_tool"
    description: str = "Reads the GNSS analysis CSV and returns summary statistics about displacement thresholds. You MUST pass the string 'read' as the argument."
    args_schema: type[BaseModel] = EmptyInput

    def _run(self, tool_input: str = "read") -> str:
        if not GNSS_ANALYSIS_PATH.exists():
            return "GNSS analysis file not found. No GNSS data available."
        
        try:
            df = pd.read_csv(GNSS_ANALYSIS_PATH)
            total_days = len(df)
            exceedances = df["threshold_exceeded"].sum() if "threshold_exceeded" in df.columns else 0
            
            summary = f"GNSS Data Summary:\n"
            summary += f"- Total observations: {total_days}\n"
            summary += f"- Days exceeding threshold: {exceedances}\n"
            
            if "displacement_mm" in df.columns:
                max_disp = df["displacement_mm"].abs().max()
                summary += f"- Maximum absolute displacement: {max_disp:.2f} mm\n"
            
            if "trend_mm_per_day" in df.columns:
                trend = df.iloc[-1]["trend_mm_per_day"]
                summary += f"- Current displacement trend: {trend:.4f} mm/day\n"
                
            return summary
        except Exception as e:
            return f"Error reading GNSS data: {str(e)}"

class ReadBridgePredictionsTool(BaseTool):
    name: str = "read_bridge_predictions_tool"
    description: str = "Reads the structural anomaly predictions across all monitored bridges. Returns a list of bridges with detected anomalies. You MUST pass the string 'read' as the argument."
    args_schema: type[BaseModel] = EmptyInput

    def _run(self, tool_input: str = "read") -> str:
        if not BRIDGE_PREDICTIONS_PATH.exists():
            return "Bridge predictions file not found. No anomaly data available."
            
        try:
            df = pd.read_csv(BRIDGE_PREDICTIONS_PATH)
            
            # We want to find bridges where anomaly == 1
            if "anomaly" not in df.columns:
                return "The predictions file is missing the 'anomaly' column."
                
            anomalies = df[df["anomaly"] == 1]
            if len(anomalies) == 0:
                return "No anomalies detected across any bridges in the current dataset."
                
            # Group by bridge to see how many anomalies per bridge
            summary = "Anomalies detected at the following bridges:\n"
            for bridge_name in anomalies["bridge_name"].unique():
                bridge_data = anomalies[anomalies["bridge_name"] == bridge_name]
                bridge_id = bridge_data.iloc[0]["bridge_id"]
                max_prob = bridge_data["anomaly_probability"].max() if "anomaly_probability" in bridge_data.columns else "N/A"
                
                count = len(bridge_data)
                summary += f"- {bridge_name} (ID: {bridge_id}): {count} anomalous readings. Max confidence: {max_prob:.2f}\n"
                
            return summary
        except Exception as e:
            return f"Error reading bridge predictions: {str(e)}"

class ReadXAIFactorsTool(BaseTool):
    name: str = "read_xai_factors_tool"
    description: str = "Reads the top explainability (XAI) factors driving the anomaly for a specific bridge ID."
    args_schema: type[BaseModel] = XAIInput

    def _run(self, bridge_id: str) -> str:
        xai_path = BRIDGES_DIR / bridge_id / "xai_top_factors.csv"
        
        if not xai_path.exists():
            return f"No XAI factors found for bridge {bridge_id}."
            
        try:
            df = pd.read_csv(xai_path)
            
            summary = f"Top root cause anomaly drivers for {bridge_id}:\n"
            for _, row in df.head(5).iterrows():
                feat = row["feature"]
                impact = row["impact"] if "impact" in row else row.get("importance_mean", 0)
                summary += f"  * {feat}: Impact score {impact:.4f}\n"
                
            return summary
        except Exception as e:
            return f"Error reading XAI factors for {bridge_id}: {str(e)}"

class ReadModelMetricsTool(BaseTool):
    name: str = "read_model_metrics_tool"
    description: str = "Reads the performance metrics of the anomaly detection ML model. You MUST pass the string 'read' as the argument."
    args_schema: type[BaseModel] = EmptyInput

    def _run(self, tool_input: str = "read") -> str:
        if not BRIDGE_MODEL_METRICS_PATH.exists():
            return "Model metrics file not found."
            
        try:
            with open(BRIDGE_MODEL_METRICS_PATH, "r") as f:
                metrics = json.load(f)
                
            summary = "Ensemble Model Performance:\n"
            summary += f"- Precision: {metrics.get('precision', 0):.3f}\n"
            summary += f"- Recall: {metrics.get('recall', 0):.3f}\n"
            summary += f"- F1-Score: {metrics.get('f1', 0):.3f}\n"
            summary += f"- PR-AUC: {metrics.get('average_precision', 0):.3f}\n"
            
            if "roc_auc" in metrics and metrics["roc_auc"] is not None:
                summary += f"- ROC-AUC: {metrics['roc_auc']:.3f}\n"
                
            return summary
        except Exception as e:
            return f"Error reading model metrics: {str(e)}"

class WriteReportTool(BaseTool):
    name: str = "write_report_tool"
    description: str = "Writes the final engineering Markdown report to the reports/ directory. Provide the full markdown content as the argument."
    args_schema: type[BaseModel] = WriteReportInput

    def _run(self, markdown_content: str) -> str:
        try:
            from datetime import datetime
            
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = reports_dir / f"shm_report_{date_str}.md"
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
                
            return f"SUCCESS: Report written to {report_path}"
        except Exception as e:
            return f"Error writing report: {str(e)}"
