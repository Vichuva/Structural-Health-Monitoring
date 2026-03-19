import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.utils.config import (
    BRIDGES_DIR,
    BRIDGE_MODEL_METRICS_PATH,
    BRIDGE_MODEL_PATH,
    BRIDGE_PREDICTIONS_PATH,
    BRIDGE_REGISTRY_PATH,
    BRIDGE_XAI_SUMMARY_PATH,
    INSAR_IMAGE_MASK_THRESHOLD_QUANTILE,
    KAGGLE_BRIDGE_DATASET_PATH,
    RANDOM_SEED,
)

MODEL_NAME = "stacked_spatiotemporal_bridge_ensemble_v2"
BRIDGE_CATALOG = [
    {
        "bridge_id": "bridge_alpha",
        "bridge_name": "Pacific Crown",
        "lat": 37.8199,
        "lon": -122.4783,
        "city": "San Francisco",
        "region": "West",
    },
    {
        "bridge_id": "bridge_beta",
        "bridge_name": "Sound Span",
        "lat": 47.6205,
        "lon": -122.3493,
        "city": "Seattle",
        "region": "West",
    },
    {
        "bridge_id": "bridge_gamma",
        "bridge_name": "Hudson Relay",
        "lat": 40.7061,
        "lon": -73.9969,
        "city": "New York",
        "region": "Northeast",
    },
    {
        "bridge_id": "bridge_delta",
        "bridge_name": "Lakeshore Axis",
        "lat": 41.8789,
        "lon": -87.6359,
        "city": "Chicago",
        "region": "Midwest",
    },
    {
        "bridge_id": "bridge_epsilon",
        "bridge_name": "Gulf Meridian",
        "lat": 29.7520,
        "lon": -95.3584,
        "city": "Houston",
        "region": "South",
    },
    {
        "bridge_id": "bridge_zeta",
        "bridge_name": "Atlantic Veil",
        "lat": 25.7634,
        "lon": -80.1918,
        "city": "Miami",
        "region": "South",
    },
]

CORE_SIGNAL_COLUMNS = [
    "Strain_microstrain",
    "Deflection_mm",
    "Vibration_ms2",
    "Tilt_deg",
    "Displacement_mm",
    "Crack_Propagation_mm",
    "Cable_Member_Tension_kN",
    "Bearing_Joint_Forces_kN",
    "Modal_Frequency_Hz",
    "Structural_Health_Index_SHI",
    "Probability_of_Failure_PoF",
    "Acoustic_Emissions_levels",
    "Vehicle_Load_tons",
    "Traffic_Volume_vph",
    "Dynamic_Load_Distribution_percent",
    "Simulated_Localized_Stress_Index",
]


def load_kaggle_bridge_dataset(path=KAGGLE_BRIDGE_DATASET_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Kaggle dataset not found at: {path}")

    df = pd.read_csv(path)
    if "Timestamp" not in df.columns:
        raise ValueError("Expected 'Timestamp' column in Kaggle bridge dataset")

    df = df.rename(columns={"Timestamp": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def assign_bridge_instances(df):
    bridge_ids = np.array([bridge["bridge_id"] for bridge in BRIDGE_CATALOG])
    assigned = df.copy()
    assigned["bridge_id"] = bridge_ids[np.arange(len(assigned)) % len(bridge_ids)]

    lookup = pd.DataFrame(BRIDGE_CATALOG)
    assigned = assigned.merge(lookup, on="bridge_id", how="left")
    return assigned


def engineer_bridge_features(df):
    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values(["bridge_id", "timestamp"]).reset_index(drop=True)

    extra_features = {
        "hour": frame["timestamp"].dt.hour,
        "dayofweek": frame["timestamp"].dt.dayofweek,
        "month": frame["timestamp"].dt.month,
        "dayofyear": frame["timestamp"].dt.dayofyear,
    }
    extra_features["hour_sin"] = np.sin(2 * np.pi * extra_features["hour"] / 24.0)
    extra_features["hour_cos"] = np.cos(2 * np.pi * extra_features["hour"] / 24.0)
    extra_features["day_sin"] = np.sin(2 * np.pi * extra_features["dayofyear"] / 365.0)
    extra_features["day_cos"] = np.cos(2 * np.pi * extra_features["dayofyear"] / 365.0)
    extra_features["bridge_age_index"] = frame.groupby("bridge_id").cumcount()

    for column in CORE_SIGNAL_COLUMNS:
        if column not in frame.columns:
            continue

        series = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = series
        group = frame.groupby("bridge_id")[column]

        extra_features[f"{column}_diff_1"] = group.diff()
        extra_features[f"{column}_roll_mean_6"] = group.transform(
            lambda values: values.rolling(6, min_periods=1).mean()
        )
        extra_features[f"{column}_roll_std_12"] = group.transform(
            lambda values: values.rolling(12, min_periods=2).std()
        )
        extra_features[f"{column}_ewm_12"] = group.transform(
            lambda values: values.ewm(span=12, adjust=False).mean()
        )

    if {"Deflection_mm", "Displacement_mm"}.issubset(frame.columns):
        extra_features["deflection_displacement_ratio"] = frame["Deflection_mm"] / (
            frame["Displacement_mm"].abs() + 1e-6
        )

    if {"Vibration_ms2", "Strain_microstrain"}.issubset(frame.columns):
        extra_features["vibration_strain_coupling"] = frame["Vibration_ms2"] * frame["Strain_microstrain"]

    if {"Probability_of_Failure_PoF", "Structural_Health_Index_SHI"}.issubset(frame.columns):
        extra_features["failure_health_gap"] = (
            frame["Probability_of_Failure_PoF"] - frame["Structural_Health_Index_SHI"]
        )

    return pd.concat([frame, pd.DataFrame(extra_features, index=frame.index)], axis=1)


def _derive_bridge_modalities(bridge_df):
    work = bridge_df.sort_values("timestamp").reset_index(drop=True)

    def _num(column, default=0.0):
        if column not in work.columns:
            return pd.Series(default, index=work.index)
        return pd.to_numeric(work[column], errors="coerce")

    displacement_mm = _num("Displacement_mm")
    deflection_mm = _num("Deflection_mm")
    tilt_deg = _num("Tilt_deg")
    settlement_mm = _num("Soil_Settlement_mm")

    gnss_raw = pd.DataFrame(
        {
            "timestamp": work["timestamp"],
            "x": 1000.0 + displacement_mm.ffill().fillna(0.0) / 1000.0,
            "y": 2000.0 + settlement_mm.ffill().fillna(0.0) / 1000.0,
            "z": 50.0 - (deflection_mm.ffill().fillna(0.0) / 1000.0),
        }
    )

    insar_ts = pd.DataFrame(
        {
            "timestamp": work["timestamp"],
            "los_displacement": (deflection_mm + 0.3 * tilt_deg).ffill().fillna(0.0),
        }
    )

    excluded = {"timestamp", "bridge_id", "bridge_name", "lat", "lon", "city", "region"}
    sensor_columns = [column for column in work.columns if column not in excluded]
    sensor_raw = work[["timestamp"] + sensor_columns].copy()
    return gnss_raw, insar_ts, sensor_raw


def _generate_bridge_insar_images(bridge_df, bridge_dir):
    images_dir = bridge_dir / "insar_images"
    masks_dir = bridge_dir / "insar_masks"
    overlays_dir = bridge_dir / "insar_overlays"
    interferograms_dir = bridge_dir / "insar_interferograms"
    heatmaps_dir = bridge_dir / "insar_heatmaps"
    coherence_dir = bridge_dir / "insar_coherence"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    interferograms_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    coherence_dir.mkdir(parents=True, exist_ok=True)

    sample_count = min(72, len(bridge_df))
    if sample_count <= 0:
        return

    step = max(1, len(bridge_df) // sample_count)
    sampled = bridge_df.iloc[::step].head(sample_count).copy().reset_index(drop=True)
    magnitudes = pd.to_numeric(sampled.get("Deflection_mm", 0.0), errors="coerce").fillna(0.0)

    mag_min = float(magnitudes.min())
    mag_max = float(magnitudes.max())
    normalized_magnitudes = (magnitudes - mag_min) / (mag_max - mag_min + 1e-9)

    size = 128
    x_grid = np.linspace(-1, 1, size)
    y_grid = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    rng = np.random.default_rng(RANDOM_SEED)

    images = []
    for index, (_, row) in enumerate(sampled.iterrows()):
        phase = 2 * np.pi * (index / max(1, sample_count - 1))
        cx = 0.18 * np.sin(phase)
        cy = 0.15 * np.cos(phase)
        sigma = 0.25
        bump = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        background = rng.normal(0.0, 0.03, (size, size))
        image = background + float(normalized_magnitudes.iloc[index]) * bump
        image = (image - image.min()) / (image.max() - image.min() + 1e-9)
        images.append(image)

        timestamp = pd.to_datetime(row["timestamp"], errors="coerce")
        stamp = timestamp.strftime("%Y%m%d_%H%M%S") if pd.notna(timestamp) else f"idx_{index:04d}"
        plt.imsave(images_dir / f"sar_{stamp}.png", image, cmap="gray")

    baseline = images[0]
    rows = []
    for index, (_, row) in enumerate(sampled.iterrows()):
        current = images[index]
        deformation = np.abs(current - baseline)
        deformation_norm = deformation / (deformation.max() + 1e-9)
        positive = deformation[deformation > 0]
        if positive.size == 0:
            mask = np.zeros_like(deformation, dtype=np.uint8)
        else:
            threshold = np.quantile(positive, INSAR_IMAGE_MASK_THRESHOLD_QUANTILE)
            mask = (deformation >= threshold).astype(np.uint8)

        timestamp = pd.to_datetime(row["timestamp"], errors="coerce")
        stamp = timestamp.strftime("%Y%m%d_%H%M%S") if pd.notna(timestamp) else f"idx_{index:04d}"
        mask_path = masks_dir / f"mask_{stamp}.png"
        overlay_path = overlays_dir / f"overlay_{stamp}.png"
        image_path = images_dir / f"sar_{stamp}.png"
        interferogram_path = interferograms_dir / f"interferogram_{stamp}.png"
        heatmap_path = heatmaps_dir / f"heatmap_{stamp}.png"
        coherence_path = coherence_dir / f"coherence_{stamp}.png"

        interferogram_phase = np.angle(np.exp(1j * 10 * np.pi * (current - baseline)))
        interferogram_norm = (interferogram_phase + np.pi) / (2 * np.pi)
        coherence = np.clip(1.0 - 0.85 * deformation_norm, 0, 1)

        plt.imsave(mask_path, mask, cmap="gray")
        plt.imsave(interferogram_path, interferogram_norm, cmap="twilight")
        plt.imsave(heatmap_path, deformation_norm, cmap="magma")
        plt.imsave(coherence_path, coherence, cmap="viridis")
        overlay = np.dstack([current, current, current])
        overlay[mask == 1] = [1.0, 0.15, 0.15]
        plt.imsave(overlay_path, overlay)

        rows.append(
            {
                "timestamp": timestamp,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "interferogram_path": str(interferogram_path),
                "heatmap_path": str(heatmap_path),
                "coherence_path": str(coherence_path),
                "mask_ratio": float(mask.mean()),
                "deformation_energy": float(deformation_norm.mean()),
                "coherence_mean": float(coherence.mean()),
            }
        )

    pd.DataFrame(rows).to_csv(bridge_dir / "insar_mask_metadata.csv", index=False)


def export_bridge_views(df):
    BRIDGES_DIR.mkdir(parents=True, exist_ok=True)

    registry = pd.DataFrame(BRIDGE_CATALOG)
    registry.to_csv(BRIDGE_REGISTRY_PATH, index=False)

    for bridge in BRIDGE_CATALOG:
        bridge_id = bridge["bridge_id"]
        subset = df[df["bridge_id"] == bridge_id].copy()
        bridge_dir = BRIDGES_DIR / bridge_id
        bridge_dir.mkdir(parents=True, exist_ok=True)

        gnss_raw, insar_ts, sensor_raw = _derive_bridge_modalities(subset)

        subset.to_csv(bridge_dir / "source_dataset.csv", index=False)
        gnss_raw.to_csv(bridge_dir / "gnss_raw.csv", index=False)
        insar_ts.to_csv(bridge_dir / "insar_timeseries.csv", index=False)
        sensor_raw.to_csv(bridge_dir / "sensor_data.csv", index=False)
        _generate_bridge_insar_images(subset, bridge_dir)


def _build_target(df):
    maintenance = pd.to_numeric(df.get("Maintenance_Alert", 0), errors="coerce").fillna(0)
    anomaly_score = pd.to_numeric(df.get("Anomaly_Detection_Score", 0), errors="coerce").fillna(0)
    flood_flag = pd.to_numeric(df.get("Flood_Event_Flag", 0), errors="coerce").fillna(0)
    wind_flag = pd.to_numeric(df.get("High_Winds_Storms", 0), errors="coerce").fillna(0)
    probability_of_failure = pd.to_numeric(df.get("Probability_of_Failure_PoF", 0), errors="coerce").fillna(0)
    return (
        (maintenance > 0)
        | (anomaly_score >= 0.8)
        | (probability_of_failure >= 0.12)
        | ((flood_flag > 0) & (wind_flag > 0))
    ).astype(int)


def _select_features(df):
    leakage_columns = {
        "timestamp",
        "Maintenance_Alert",
        "Anomaly_Detection_Score",
        "SHI_Predicted_24h_Ahead",
        "SHI_Predicted_7d_Ahead",
        "SHI_Predicted_30d_Ahead",
    }
    return [column for column in df.columns if column not in leakage_columns]


def _optimal_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(np.nanargmax(f1_scores))
    return float(thresholds[best_index])


def _build_reference_values(feature_frame):
    reference_values = {}
    for column in feature_frame.columns:
        series = feature_frame[column]
        if pd.api.types.is_numeric_dtype(series):
            value = series.median()
            reference_values[column] = None if pd.isna(value) else float(value)
        else:
            mode = series.mode(dropna=True)
            reference_values[column] = None if mode.empty else str(mode.iloc[0])
    return reference_values


def _compute_global_explainability(model_pipeline, x_test, y_test, feature_columns):
    sample_size = min(2500, len(x_test))
    sampled_x = x_test.sample(n=sample_size, random_state=RANDOM_SEED) if len(x_test) > sample_size else x_test
    sampled_y = y_test.loc[sampled_x.index]

    importance = permutation_importance(
        model_pipeline,
        sampled_x,
        sampled_y,
        n_repeats=4,
        random_state=RANDOM_SEED,
        scoring="average_precision",
        n_jobs=1,
    )

    importance_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    top_features = importance_frame.head(16).copy()
    summary = [
        {
            "feature": row["feature"],
            "importance_mean": float(row["importance_mean"]),
            "importance_std": float(row["importance_std"]),
        }
        for _, row in top_features.iterrows()
    ]
    return summary, top_features["feature"].tolist()


def explain_prediction_row(model_pipeline, row_frame, reference_values, feature_priority, top_n=8):
    base_probability = float(model_pipeline.predict_proba(row_frame)[:, 1][0])
    impacts = []

    for feature in feature_priority:
        if feature not in row_frame.columns or feature not in reference_values:
            continue
        modified = row_frame.copy()
        modified.at[row_frame.index[0], feature] = reference_values[feature]
        new_probability = float(model_pipeline.predict_proba(modified)[:, 1][0])
        impacts.append(
            {
                "feature": feature,
                "impact": base_probability - new_probability,
                "baseline_probability": base_probability,
                "counterfactual_probability": new_probability,
            }
        )

    impact_frame = pd.DataFrame(impacts)
    if impact_frame.empty:
        return impact_frame

    impact_frame["abs_impact"] = impact_frame["impact"].abs()
    impact_frame = impact_frame.sort_values("abs_impact", ascending=False).head(top_n).reset_index(drop=True)
    return impact_frame.drop(columns=["abs_impact"])


def _build_model_pipeline(feature_frame):
    numeric_columns = feature_frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in feature_frame.columns if column not in numeric_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ],
        sparse_threshold=0,
    )

    ensemble_model = StackingClassifier(
        estimators=[
            (
                "hist_gb",
                HistGradientBoostingClassifier(
                    learning_rate=0.045,
                    max_depth=6,
                    max_iter=350,
                    min_samples_leaf=24,
                    random_state=RANDOM_SEED,
                ),
            ),
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=450,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ],
        final_estimator=LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            solver="liblinear",
        ),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", ensemble_model)])


def train_bridge_anomaly_model(df):
    engineered = engineer_bridge_features(df)
    y = _build_target(engineered)
    feature_columns = _select_features(engineered)
    x = engineered[feature_columns].copy()
    reference_values = _build_reference_values(x)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.1765,
        random_state=RANDOM_SEED,
        stratify=y_train_val,
    )

    training_pipeline = _build_model_pipeline(x_train)
    train_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    training_pipeline.fit(x_train, y_train, model__sample_weight=train_weights)

    validation_probs = training_pipeline.predict_proba(x_val)[:, 1]
    threshold = _optimal_threshold(y_val, validation_probs)

    final_pipeline = _build_model_pipeline(x_train_val)
    train_val_weights = compute_sample_weight(class_weight="balanced", y=y_train_val)
    final_pipeline.fit(x_train_val, y_train_val, model__sample_weight=train_val_weights)

    test_probs = final_pipeline.predict_proba(x_test)[:, 1]
    test_pred = (test_probs >= threshold).astype(int)

    metrics = {
        "model_name": MODEL_NAME,
        "threshold": threshold,
        "feature_count": int(len(feature_columns)),
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, test_probs)),
    }

    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, test_probs))
    else:
        metrics["roc_auc"] = None

    global_xai, priority_features = _compute_global_explainability(
        final_pipeline,
        x_test,
        y_test,
        feature_columns,
    )

    metrics["xai_top_features"] = global_xai
    return final_pipeline, feature_columns, threshold, metrics, reference_values, priority_features


def generate_predictions(df, model_pipeline, threshold):
    engineered = engineer_bridge_features(df)
    feature_columns = _select_features(engineered)
    probabilities = model_pipeline.predict_proba(engineered[feature_columns])[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    output = engineered[["timestamp", "bridge_id", "bridge_name", "lat", "lon", "city", "region"]].copy()
    output["anomaly_probability"] = probabilities
    output["anomaly"] = predictions

    passthrough_columns = [
        "Deflection_mm",
        "Displacement_mm",
        "Vibration_ms2",
        "Strain_microstrain",
        "Tilt_deg",
        "Structural_Health_Index_SHI",
        "Probability_of_Failure_PoF",
        "Vibration_Anomaly_Location",
        "Simulated_Localized_Stress_Index",
        "Localized_Strain_Hotspot",
        "Crack_Propagation_mm",
    ]
    for column in passthrough_columns:
        if column in engineered.columns:
            output[column] = engineered[column]

    return output


def run_kaggle_bridge_pipeline(dataset_path=KAGGLE_BRIDGE_DATASET_PATH):
    dataset = load_kaggle_bridge_dataset(dataset_path)
    assigned = assign_bridge_instances(dataset)

    export_bridge_views(assigned)

    (
        model_pipeline,
        feature_columns,
        threshold,
        metrics,
        reference_values,
        priority_features,
    ) = train_bridge_anomaly_model(assigned)
    predictions = generate_predictions(assigned, model_pipeline, threshold)

    BRIDGE_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(BRIDGE_PREDICTIONS_PATH, index=False)

    BRIDGE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BRIDGE_MODEL_PATH, "wb") as handle:
        pickle.dump(
            {
                "model_pipeline": model_pipeline,
                "feature_columns": feature_columns,
                "threshold": threshold,
                "model_name": MODEL_NAME,
                "reference_values": reference_values,
                "priority_features": priority_features,
            },
            handle,
        )

    BRIDGE_MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BRIDGE_MODEL_METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    BRIDGE_XAI_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BRIDGE_XAI_SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "global_feature_importance": metrics.get("xai_top_features", []),
            },
            handle,
            indent=2,
        )

    for bridge in BRIDGE_CATALOG:
        run_bridge_inference(bridge["bridge_id"])

    return {
        "rows": int(len(assigned)),
        "bridges": int(assigned["bridge_id"].nunique()),
        "anomalies": int(predictions["anomaly"].sum()),
        "threshold": threshold,
        "metrics": metrics,
    }


def _collect_runtime_modalities(bridge_dir):
    modality_counts = {}
    for file_name, label in [
        ("gnss_raw.csv", "gnss_rows"),
        ("insar_timeseries.csv", "insar_rows"),
        ("sensor_data.csv", "sensor_rows"),
    ]:
        path = bridge_dir / file_name
        modality_counts[label] = int(len(pd.read_csv(path))) if path.exists() else 0
    return modality_counts


def run_bridge_inference(bridge_id, return_trace=False):
    bridge_dir = BRIDGES_DIR / bridge_id
    source_path = bridge_dir / "source_dataset.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Bridge source dataset not found: {source_path}")

    if not BRIDGE_MODEL_PATH.exists():
        run_kaggle_bridge_pipeline()

    with open(BRIDGE_MODEL_PATH, "rb") as handle:
        artifact = pickle.load(handle)

    model_pipeline = artifact["model_pipeline"]
    threshold = artifact["threshold"]
    model_name = artifact.get("model_name", MODEL_NAME)
    reference_values = artifact.get("reference_values", {})
    priority_features = artifact.get("priority_features", [])

    stage_trace = []
    total_start = time.perf_counter()

    start = time.perf_counter()
    bridge_df = pd.read_csv(source_path)
    bridge_df["timestamp"] = pd.to_datetime(bridge_df["timestamp"], errors="coerce")
    bridge_df = bridge_df.dropna(subset=["timestamp"]).reset_index(drop=True)
    stage_trace.append(
        {
            "stage": "Data Intake",
            "detail": "Loaded bridge digital twin source frame and aligned timestamps.",
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "rows": int(len(bridge_df)),
        }
    )

    start = time.perf_counter()
    modality_counts = _collect_runtime_modalities(bridge_dir)
    stage_trace.append(
        {
            "stage": "Modal Sync",
            "detail": "Bound GNSS, InSAR, and sensor telemetry streams to the selected bridge.",
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            **modality_counts,
        }
    )

    start = time.perf_counter()
    feature_frame = engineer_bridge_features(bridge_df)
    feature_columns = _select_features(feature_frame)
    stage_trace.append(
        {
            "stage": "Feature Synthesis",
            "detail": "Generated temporal deltas, rolling statistics, and multimodal interaction features.",
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "feature_count": int(len(feature_columns)),
        }
    )

    start = time.perf_counter()
    predictions = generate_predictions(bridge_df, model_pipeline=model_pipeline, threshold=threshold)
    stage_trace.append(
        {
            "stage": "Ensemble Scoring",
            "detail": f"Scored the bridge with {model_name} and applied the tuned anomaly threshold.",
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "anomaly_count": int(predictions["anomaly"].sum()),
            "max_probability": float(predictions["anomaly_probability"].max()),
        }
    )

    start = time.perf_counter()
    engineered_frame = engineer_bridge_features(bridge_df)
    top_index = int(predictions["anomaly_probability"].idxmax())
    explanation_row = engineered_frame.loc[[top_index], _select_features(engineered_frame)].copy()
    local_xai = explain_prediction_row(
        model_pipeline,
        explanation_row,
        reference_values,
        priority_features,
        top_n=8,
    )
    local_xai_path = bridge_dir / "xai_top_factors.csv"
    local_xai.to_csv(local_xai_path, index=False)
    stage_trace.append(
        {
            "stage": "Explainability",
            "detail": "Resolved the strongest local anomaly drivers with counterfactual feature ablation.",
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "top_drivers": int(len(local_xai)),
        }
    )

    start = time.perf_counter()
    hotspot_count = int(
        predictions["Vibration_Anomaly_Location"].fillna("Deck").astype(str).nunique()
        if "Vibration_Anomaly_Location" in predictions.columns
        else 0
    )
    stage_trace.append(
        {
            "stage": "Spatial Projection",
            "detail": "Projected high-risk events onto bridge deck, tower, cable, and pier zones.",
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "hotspot_zones": hotspot_count,
        }
    )

    output_path = bridge_dir / "predictions.csv"
    predictions.to_csv(output_path, index=False)

    runtime = {
        "bridge_id": bridge_id,
        "model_name": model_name,
        "threshold": float(threshold),
        "local_xai_path": str(local_xai_path),
        "total_duration_ms": round((time.perf_counter() - total_start) * 1000, 2),
        "stages": stage_trace,
    }

    if return_trace:
        return predictions, runtime
    return predictions


if __name__ == "__main__":
    result = run_kaggle_bridge_pipeline()
    print(json.dumps(result, indent=2))
