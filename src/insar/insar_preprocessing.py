from pathlib import Path

import pandas as pd

from src.utils.config import INSAR_PROCESSED_PATH


def load_insar_timeseries(path):
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("InSAR data requires a 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    return df.sort_values("timestamp").reset_index(drop=True)


def normalize_displacement(df):
    normalized = df.copy()

    if "los_displacement" in normalized.columns:
        normalized["los_disp_norm"] = (
            normalized["los_displacement"] - normalized["los_displacement"].iloc[0]
        )
    elif "los_disp_norm" not in normalized.columns:
        raise ValueError("InSAR data requires either 'los_displacement' or 'los_disp_norm'")

    return normalized[["timestamp", "los_disp_norm"]]


def run_insar_preprocessing(input_path=INSAR_PROCESSED_PATH, output_path=INSAR_PROCESSED_PATH):
    insar = load_insar_timeseries(input_path)
    normalized = normalize_displacement(insar)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)
    return normalized


if __name__ == "__main__":
    run_insar_preprocessing()
    print("InSAR preprocessing completed")
