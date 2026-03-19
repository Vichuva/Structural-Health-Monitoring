from pathlib import Path

import pandas as pd

from src.utils.config import (
    FUSED_OUTPUT_PATH,
    FUSION_WEIGHT_GNSS,
    FUSION_WEIGHT_INSAR,
    GNSS_PROCESSED_PATH,
    INSAR_PROCESSED_PATH,
)


def fuse_data(gnss_path=GNSS_PROCESSED_PATH, insar_path=INSAR_PROCESSED_PATH):
    gnss = pd.read_csv(gnss_path)
    insar = pd.read_csv(insar_path)

    gnss_required = {"timestamp", "dz"}
    insar_required = {"timestamp", "los_disp_norm"}

    gnss_missing = gnss_required.difference(gnss.columns)
    insar_missing = insar_required.difference(insar.columns)

    if gnss_missing:
        raise ValueError(f"GNSS displacement data missing columns: {sorted(gnss_missing)}")
    if insar_missing:
        raise ValueError(f"InSAR data missing columns: {sorted(insar_missing)}")

    gnss["timestamp"] = pd.to_datetime(gnss["timestamp"], errors="raise")
    insar["timestamp"] = pd.to_datetime(insar["timestamp"], errors="raise")

    fused = pd.merge_asof(
        gnss.sort_values("timestamp"),
        insar.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
    )

    fused["gnss_dz_mm"] = fused["dz"] * 1000.0
    fused["fused_disp"] = (
        FUSION_WEIGHT_GNSS * fused["gnss_dz_mm"]
        + FUSION_WEIGHT_INSAR * fused["los_disp_norm"]
    )

    return fused[["timestamp", "gnss_dz_mm", "los_disp_norm", "fused_disp"]]


def run_data_fusion(
    gnss_path=GNSS_PROCESSED_PATH,
    insar_path=INSAR_PROCESSED_PATH,
    output_path=FUSED_OUTPUT_PATH,
):
    fused = fuse_data(gnss_path=gnss_path, insar_path=insar_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(output_path, index=False)
    return fused


if __name__ == "__main__":
    run_data_fusion()
    print("Data fusion completed")
