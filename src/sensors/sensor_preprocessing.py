from pathlib import Path

import pandas as pd

from src.utils.config import SENSOR_PROCESSED_PATH, SENSOR_RAW_PATH


def load_sensor_data(path):
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Sensor data requires a 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_sensor_features(df):
    feature_df = df.copy()

    numeric_cols = [c for c in feature_df.columns if c != "timestamp"]
    for col in numeric_cols:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    feature_df[numeric_cols] = feature_df[numeric_cols].interpolate(limit_direction="both")
    feature_df[numeric_cols] = feature_df[numeric_cols].fillna(feature_df[numeric_cols].median())

    if "vibration_rms" in feature_df.columns:
        feature_df["vibration_rms_roll3"] = feature_df["vibration_rms"].rolling(3, min_periods=1).mean()
    if "strain_ue" in feature_df.columns:
        feature_df["strain_gradient"] = feature_df["strain_ue"].diff().fillna(0.0)
    if "acoustic_db" in feature_df.columns:
        feature_df["acoustic_roll3"] = feature_df["acoustic_db"].rolling(3, min_periods=1).mean()

    return feature_df


def run_sensor_preprocessing(input_path=SENSOR_RAW_PATH, output_path=SENSOR_PROCESSED_PATH):
    sensors = load_sensor_data(input_path)
    features = build_sensor_features(sensors)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return features


if __name__ == "__main__":
    run_sensor_preprocessing()
    print("Sensor preprocessing completed")
