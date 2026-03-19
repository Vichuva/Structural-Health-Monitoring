import argparse

from src.anomaly_detection.detect_anomalies import run_anomaly_detection
from src.bridges.kaggle_bridge_pipeline import run_kaggle_bridge_pipeline
from src.fusion.data_fusion import run_data_fusion
from src.gnss.gnss_analysis import run_gnss_analysis
from src.gnss.gnss_preprocessing import run_gnss_preprocessing
from src.insar.insar_analysis import run_insar_analysis
from src.insar.insar_image_processing import process_insar_images
from src.insar.insar_preprocessing import run_insar_preprocessing
from src.sensors.sensor_preprocessing import run_sensor_preprocessing
from src.utils.generate_synthetic_gnss import generate_synthetic_gnss
from src.utils.generate_synthetic_insar import generate_synthetic_insar
from src.utils.generate_synthetic_sensor_data import generate_synthetic_sensor_data


def run_pipeline(generate_synthetic_data=True):
    if generate_synthetic_data:
        generate_synthetic_gnss()
        generate_synthetic_insar()
        generate_synthetic_sensor_data()

    gnss_disp = run_gnss_preprocessing()
    insar_norm = run_insar_preprocessing()
    sensor_features = run_sensor_preprocessing()

    gnss_analysis = run_gnss_analysis()
    insar_analysis = run_insar_analysis()

    fused = run_data_fusion()
    anomalies = run_anomaly_detection()
    insar_masks = process_insar_images()

    summary = {
        "gnss_rows": len(gnss_disp),
        "insar_rows": len(insar_norm),
        "sensor_rows": len(sensor_features),
        "fused_rows": len(fused),
        "anomaly_count": int(anomalies["anomaly"].sum()) if "anomaly" in anomalies.columns else 0,
        "gnss_threshold_hits": int(gnss_analysis["threshold_exceeded"].sum()),
        "insar_threshold_hits": int(insar_analysis["threshold_exceeded"].sum()),
        "insar_image_frames": len(insar_masks),
    }
    return summary


def run_all_pipelines(generate_synthetic_data=True, run_kaggle=True):
    summary = {"synthetic": run_pipeline(generate_synthetic_data=generate_synthetic_data)}
    if run_kaggle:
        summary["kaggle"] = run_kaggle_bridge_pipeline()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Structural health monitoring pipeline")
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip synthetic data generation and use existing input files",
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Skip Kaggle bridge model training pipeline",
    )
    args = parser.parse_args()

    summary = run_all_pipelines(
        generate_synthetic_data=not args.skip_generate,
        run_kaggle=not args.skip_kaggle,
    )
    synthetic = summary["synthetic"]

    print("Structural Health Monitoring pipeline completed")
    print(
        "Rows: "
        f"GNSS={synthetic['gnss_rows']}, "
        f"InSAR={synthetic['insar_rows']}, "
        f"Sensors={synthetic['sensor_rows']}, "
        f"Fused={synthetic['fused_rows']}"
    )
    print(
        "Flags: "
        f"GNSS threshold hits={synthetic['gnss_threshold_hits']}, "
        f"InSAR threshold hits={synthetic['insar_threshold_hits']}, "
        f"Detected anomalies={synthetic['anomaly_count']}, "
        f"InSAR frames={synthetic['insar_image_frames']}"
    )
    if "kaggle" in summary:
        kaggle = summary["kaggle"]
        print(
            "Kaggle model: "
            f"rows={kaggle['rows']}, "
            f"bridges={kaggle['bridges']}, "
            f"predicted_anomalies={kaggle['anomalies']}, "
            f"threshold={kaggle['threshold']:.4f}"
        )
        metrics = kaggle["metrics"]
        print(
            "Kaggle metrics: "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"pr_auc={metrics['average_precision']:.4f}"
        )


if __name__ == "__main__":
    main()
