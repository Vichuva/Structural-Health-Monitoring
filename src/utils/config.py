from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
BRIDGES_DIR = DATA_DIR / "bridges"

GNSS_RAW_PATH = DATA_DIR / "gnss" / "raw" / "gnss_raw.csv"
GNSS_PROCESSED_PATH = DATA_DIR / "gnss" / "processed" / "gnss_displacement.csv"
GNSS_ANALYSIS_PATH = DATA_DIR / "gnss" / "processed" / "gnss_analysis.csv"

INSAR_RAW_IMAGE_DIR = DATA_DIR / "insar" / "raw" / "sentinel_images"
INSAR_PROCESSED_PATH = DATA_DIR / "insar" / "processed" / "insar_timeseries.csv"
INSAR_ANALYSIS_PATH = DATA_DIR / "insar" / "processed" / "insar_analysis.csv"
INSAR_MASK_DIR = DATA_DIR / "insar" / "processed" / "masks"
INSAR_OVERLAY_DIR = DATA_DIR / "insar" / "processed" / "overlays"
INSAR_MASK_METADATA_PATH = DATA_DIR / "insar" / "processed" / "insar_mask_metadata.csv"

SENSOR_RAW_PATH = DATA_DIR / "sensors" / "raw" / "sensor_data.csv"
SENSOR_PROCESSED_PATH = DATA_DIR / "sensors" / "processed" / "sensor_features.csv"

FUSED_OUTPUT_PATH = DATA_DIR / "fused" / "fused_displacement.csv"
FUSED_TRAINING_FRAME_PATH = DATA_DIR / "fused" / "fused_training_frame.csv"
SENSOR_ANOMALY_MODEL_PATH = MODELS_DIR / "sensor_anomaly_model.pkl"

KAGGLE_BRIDGE_DATASET_PATH = EXTERNAL_DATA_DIR / "bridge_digital_twin_dataset.csv"
BRIDGE_REGISTRY_PATH = BRIDGES_DIR / "bridge_registry.csv"
BRIDGE_PREDICTIONS_PATH = BRIDGES_DIR / "bridge_predictions.csv"
BRIDGE_MODEL_PATH = MODELS_DIR / "bridge_anomaly_model.pkl"
BRIDGE_MODEL_METRICS_PATH = MODELS_DIR / "bridge_anomaly_metrics.json"
BRIDGE_XAI_SUMMARY_PATH = MODELS_DIR / "bridge_xai_summary.json"

# Domain thresholds
GNSS_THRESHOLD_MM = 5
INSAR_THRESHOLD_MM = 8
ANOMALY_SCORE_THRESHOLD = 0.7

# Fusion and anomaly parameters
FUSION_WEIGHT_GNSS = 0.6
FUSION_WEIGHT_INSAR = 0.4
ISOLATION_FOREST_CONTAMINATION = 0.1
RANDOM_SEED = 42
INSAR_IMAGE_MASK_THRESHOLD_QUANTILE = 0.97
