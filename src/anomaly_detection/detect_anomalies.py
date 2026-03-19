from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils.config import (
    ANOMALY_SCORE_THRESHOLD,
    FUSED_OUTPUT_PATH,
    FUSED_TRAINING_FRAME_PATH,
    GNSS_ANALYSIS_PATH,
    INSAR_ANALYSIS_PATH,
    ISOLATION_FOREST_CONTAMINATION,
    RANDOM_SEED,
    SENSOR_ANOMALY_MODEL_PATH,
    SENSOR_PROCESSED_PATH,
)


def _prepare_frame(df):
    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _prefix_columns(df, prefix):
    renamed = df.copy()
    renamed.columns = ["timestamp" if c == "timestamp" else f"{prefix}{c}" for c in renamed.columns]
    return renamed


def build_training_frame(fused_df, sensor_df=None, gnss_analysis_df=None, insar_analysis_df=None):
    if "timestamp" not in fused_df.columns:
        raise ValueError("Fused dataframe requires a 'timestamp' column")

    merged = _prepare_frame(fused_df)

    aux_frames = [
        (sensor_df, "sensor_"),
        (gnss_analysis_df, "gnss_"),
        (insar_analysis_df, "insar_"),
    ]

    for aux_df, prefix in aux_frames:
        if aux_df is None:
            continue
        if aux_df.empty or "timestamp" not in aux_df.columns:
            continue

        aux = _prefix_columns(_prepare_frame(aux_df), prefix)
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            aux.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        )

    numeric_cols = [c for c in merged.columns if c != "timestamp"]
    merged[numeric_cols] = merged[numeric_cols].apply(pd.to_numeric, errors="coerce")
    merged[numeric_cols] = merged[numeric_cols].interpolate(limit_direction="both")
    merged[numeric_cols] = merged[numeric_cols].fillna(merged[numeric_cols].median())

    return merged


def train_and_predict_anomalies(training_df):
    if training_df.empty:
        training_df["anomaly_score"] = []
        training_df["anomaly"] = []
        return training_df, []

    feature_cols = [c for c in training_df.columns if c != "timestamp"]
    x = training_df[feature_cols]

    if len(training_df) < 5:
        out = training_df.copy()
        out["anomaly_score"] = 0.0
        out["anomaly"] = 0
        return out, feature_cols

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    contamination = min(max(ISOLATION_FOREST_CONTAMINATION, 0.001), 0.49)
    model = IsolationForest(contamination=contamination, random_state=RANDOM_SEED)

    labels = model.fit_predict(x_scaled)
    raw_scores = -model.decision_function(x_scaled)

    score_min = float(raw_scores.min())
    score_max = float(raw_scores.max())
    if score_max == score_min:
        normalized_scores = np.zeros_like(raw_scores)
    else:
        normalized_scores = (raw_scores - score_min) / (score_max - score_min)

    out = training_df.copy()
    out["anomaly_score"] = normalized_scores
    out["anomaly"] = ((labels == -1) | (out["anomaly_score"] >= ANOMALY_SCORE_THRESHOLD)).astype(int)

    model_artifact = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "score_min": score_min,
        "score_max": score_max,
    }
    return out, model_artifact


def _load_optional_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def run_anomaly_detection(
    fused_path=FUSED_OUTPUT_PATH,
    sensor_path=SENSOR_PROCESSED_PATH,
    gnss_analysis_path=GNSS_ANALYSIS_PATH,
    insar_analysis_path=INSAR_ANALYSIS_PATH,
    output_path=FUSED_OUTPUT_PATH,
    training_frame_output=FUSED_TRAINING_FRAME_PATH,
    model_output=SENSOR_ANOMALY_MODEL_PATH,
):
    fused_df = pd.read_csv(fused_path)
    if "fused_disp" not in fused_df.columns:
        raise ValueError("Fused data missing required 'fused_disp' column")

    sensor_df = _load_optional_csv(sensor_path)
    gnss_df = _load_optional_csv(gnss_analysis_path)
    insar_df = _load_optional_csv(insar_analysis_path)

    training_df = build_training_frame(
        fused_df=fused_df,
        sensor_df=sensor_df,
        gnss_analysis_df=gnss_df,
        insar_analysis_df=insar_df,
    )

    predicted_df, model_artifact = train_and_predict_anomalies(training_df)

    Path(training_frame_output).parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(training_frame_output, index=False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predicted_df.to_csv(output_path, index=False)

    if isinstance(model_artifact, dict):
        Path(model_output).parent.mkdir(parents=True, exist_ok=True)
        with open(model_output, "wb") as f:
            pickle.dump(model_artifact, f)

    return predicted_df


if __name__ == "__main__":
    run_anomaly_detection()
    print("Sensor-aware anomaly detection completed")
