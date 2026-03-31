from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import GNSS_ANALYSIS_PATH, GNSS_PROCESSED_PATH, GNSS_THRESHOLD_MM


def analyze_gnss_displacement(df):
    analysis = df.copy()

    # Ensure proper time order
    analysis["timestamp"] = pd.to_datetime(analysis["timestamp"])
    analysis = analysis.sort_values("timestamp")

    # Compute displacements
    analysis["horizontal_mm"] = np.sqrt(analysis["dx"]**2 + analysis["dy"]**2) * 1000
    analysis["vertical_mm"] = analysis["dz"] * 1000
    analysis["total_mm"] = np.sqrt(
        analysis["dx"]**2 + analysis["dy"]**2 + analysis["dz"]**2
    ) * 1000

    # Rolling smoothing (VERY IMPORTANT for visualization)
    analysis["horizontal_mm_smooth"] = analysis["horizontal_mm"].rolling(window=5, min_periods=1).mean()
    analysis["vertical_mm_smooth"] = analysis["vertical_mm"].rolling(window=5, min_periods=1).mean()
    analysis["total_mm_smooth"] = analysis["total_mm"].rolling(window=5, min_periods=1).mean()

    # Normalize (to avoid flat visualization if values are too small)
    analysis["total_mm_norm"] = (
        (analysis["total_mm"] - analysis["total_mm"].min()) /
        (analysis["total_mm"].max() - analysis["total_mm"].min() + 1e-9)
    )

    # Threshold detection
    analysis["threshold_exceeded"] = (
        analysis["vertical_mm"].abs() >= GNSS_THRESHOLD_MM
    ).astype(int)

    return analysis[
        [
            "timestamp",
            "horizontal_mm",
            "vertical_mm",
            "total_mm",
            "horizontal_mm_smooth",
            "vertical_mm_smooth",
            "total_mm_smooth",
            "total_mm_norm",
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
