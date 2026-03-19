from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import GNSS_ANALYSIS_PATH, GNSS_PROCESSED_PATH, GNSS_THRESHOLD_MM


def analyze_gnss_displacement(df):
    analysis = df.copy()
    analysis["horizontal_mm"] = np.sqrt(analysis["dx"] ** 2 + analysis["dy"] ** 2) * 1000.0
    analysis["vertical_mm"] = analysis["dz"] * 1000.0
    analysis["total_mm"] = np.sqrt(
        analysis["dx"] ** 2 + analysis["dy"] ** 2 + analysis["dz"] ** 2
    ) * 1000.0
    analysis["threshold_exceeded"] = (
        analysis["vertical_mm"].abs() >= GNSS_THRESHOLD_MM
    ).astype(int)

    return analysis[
        [
            "timestamp",
            "horizontal_mm",
            "vertical_mm",
            "total_mm",
            "threshold_exceeded",
        ]
    ]


def run_gnss_analysis(input_path=GNSS_PROCESSED_PATH, output_path=GNSS_ANALYSIS_PATH):
    df = pd.read_csv(input_path)
    required = {"timestamp", "dx", "dy", "dz"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"GNSS displacement data missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    analysis = analyze_gnss_displacement(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(output_path, index=False)
    return analysis


if __name__ == "__main__":
    run_gnss_analysis()
    print("GNSS analysis completed")
