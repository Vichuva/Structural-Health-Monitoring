from pathlib import Path

import pandas as pd

from src.utils.config import INSAR_ANALYSIS_PATH, INSAR_PROCESSED_PATH, INSAR_THRESHOLD_MM


def analyze_insar_displacement(df):
    analysis = df.copy()
    analysis["abs_los_disp"] = analysis["los_disp_norm"].abs()
    analysis["threshold_exceeded"] = (
        analysis["abs_los_disp"] >= INSAR_THRESHOLD_MM
    ).astype(int)
    return analysis[["timestamp", "los_disp_norm", "abs_los_disp", "threshold_exceeded"]]


def run_insar_analysis(input_path=INSAR_PROCESSED_PATH, output_path=INSAR_ANALYSIS_PATH):
    df = pd.read_csv(input_path)
    required = {"timestamp", "los_disp_norm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"InSAR processed data missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    analysis = analyze_insar_displacement(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(output_path, index=False)
    return analysis


if __name__ == "__main__":
    run_insar_analysis()
    print("InSAR analysis completed")
