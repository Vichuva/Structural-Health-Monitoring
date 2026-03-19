from pathlib import Path

import pandas as pd

from src.utils.config import GNSS_PROCESSED_PATH, GNSS_RAW_PATH

REQUIRED_COLUMNS = {"timestamp", "x", "y", "z"}


def load_gnss_data(path):
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"GNSS data missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_displacement(df):
    ref = df.iloc[0]
    displacement = df.copy()
    displacement["dx"] = displacement["x"] - ref["x"]
    displacement["dy"] = displacement["y"] - ref["y"]
    displacement["dz"] = displacement["z"] - ref["z"]
    return displacement[["timestamp", "dx", "dy", "dz"]]


def run_gnss_preprocessing(input_path=GNSS_RAW_PATH, output_path=GNSS_PROCESSED_PATH):
    gnss = load_gnss_data(input_path)
    disp = compute_displacement(gnss)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    disp.to_csv(output_path, index=False)
    return disp


if __name__ == "__main__":
    run_gnss_preprocessing()
    print("GNSS preprocessing completed")
