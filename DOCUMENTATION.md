# Structural Health Monitoring — Bridge Digital Twin
## Complete Technical Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Configuration — `src/utils/config.py`](#4-configuration)
5. [Synthetic Data Generators](#5-synthetic-data-generators)
   - 5.1 GNSS Data Generator
   - 5.2 InSAR Data Generator
   - 5.3 Sensor Data Generator
6. [GNSS Module](#6-gnss-module)
   - 6.1 Preprocessing
   - 6.2 Analysis
7. [InSAR Module](#7-insar-module)
   - 7.1 Preprocessing
   - 7.2 Analysis
   - 7.3 Image Processing
8. [Sensor Module](#8-sensor-module)
9. [Data Fusion Module](#9-data-fusion-module)
10. [Anomaly Detection Module](#10-anomaly-detection-module)
11. [Bridge (Kaggle) Pipeline](#11-bridge-kaggle-pipeline)
12. [Visualization Dashboard](#12-visualization-dashboard)
13. [Entry Point — `main.py`](#13-entry-point--mainpy)
14. [Data Flow Diagram](#14-data-flow-diagram)
15. [Key Thresholds and Parameters](#15-key-thresholds-and-parameters)
16. [Generated Outputs](#16-generated-outputs)
17. [Setup and Running the Project](#17-setup-and-running-the-project)
18. [Dependencies](#18-dependencies)
19. [Glossary](#19-glossary)

---

## 1. Project Overview

**Structural Health Monitoring (SHM)** is an end-to-end Python system designed to monitor the structural integrity of bridges using multiple data sources. It combines satellite geodesy, radar remote sensing, IoT sensor streams, and machine learning to detect anomalies and visualize structural health in real-time through a web-based dashboard.

### What It Does

The system integrates three independent sensing modalities:

| Modality | Technology | Measures |
|---|---|---|
| **GNSS** | GPS/GNSS satellite positioning | Ground displacement in X, Y, Z (millimetres) |
| **InSAR** | SAR (Synthetic Aperture Radar) satellite imagery | Line-of-sight (LOS) surface deformation |
| **Sensors** | IoT sensors on the bridge | Vibration, strain, tilt, temperature, acoustic emissions |

These are fused together and fed into two machine learning models:
1. **Isolation Forest** — unsupervised anomaly detection on the fused synthetic sensor data
2. **Stacked Ensemble Classifier** — supervised anomaly classification trained on the Kaggle bridge digital twin dataset

Results are displayed on an interactive **Streamlit dashboard** with map-based bridge selection, 3D visualizations, InSAR image overlays, and live anomaly tables.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────┐
│                    DATA SOURCES                       │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐ │
│  │  GNSS    │  │  InSAR   │  │  IoT Sensors        │ │
│  │(satellite│  │(satellite│  │(vibration, strain,  │ │
│  │ GPS data)│  │  radar)  │  │ tilt, acoustic, etc)│ │
│  └────┬─────┘  └────┬─────┘  └──────────┬──────────┘ │
└───────┼─────────────┼────────────────────┼────────────┘
        │             │                    │
        ▼             ▼                    ▼
┌───────────┐  ┌────────────┐     ┌─────────────────┐
│  GNSS     │  │  InSAR     │     │    Sensor       │
│ Preprocess│  │ Preprocess │     │  Preprocessing  │
│(displacement│  │(normalize │     │ (feature eng.) │
│   calc.)  │  │   LOS)    │     │                 │
└─────┬─────┘  └─────┬──────┘     └────────┬────────┘
      │               │                     │
      ├───────────────┘                     │
      ▼                                     │
┌─────────────┐                             │
│ Data Fusion │◄────────────────────────────┘
│(GNSS + InSAR│
│  weighted   │
│   merge)    │
└──────┬──────┘
       │
       ▼
┌──────────────────┐       ┌──────────────────────────┐
│ Anomaly Detection│       │  Kaggle Bridge Pipeline  │
│(Isolation Forest)│       │(Stacked Ensemble Model)  │
└──────┬───────────┘       └──────────┬───────────────┘
       │                              │
       └──────────────┬───────────────┘
                      ▼
             ┌────────────────┐
             │   Streamlit    │
             │   Dashboard    │
             │ (visualization)│
             └────────────────┘
```

---

## 3. Directory Structure

```
Structural-Health-Monitoring/
│
├── main.py                        ← Entry point; orchestrates the full pipeline
├── requirements.txt               ← Python package dependencies
├── pyrightconfig.json             ← Pyright/Pylance type checker config
├── .pyre_configuration            ← Pyre2 type checker config
├── README.md                      ← Brief project readme
├── DOCUMENTATION.md               ← This file
│
├── src/                           ← All source code
│   ├── __init__.py
│   ├── utils/
│   │   ├── config.py              ← All paths, thresholds, and hyperparameters
│   │   ├── generate_synthetic_gnss.py
│   │   ├── generate_synthetic_insar.py
│   │   └── generate_synthetic_sensor_data.py
│   │
│   ├── gnss/
│   │   ├── gnss_preprocessing.py  ← Loads + computes displacement from raw GNSS
│   │   └── gnss_analysis.py       ← Computes horizontal/vertical/total mm + flags
│   │
│   ├── insar/
│   │   ├── insar_preprocessing.py ← Normalizes LOS displacement timeseries
│   │   ├── insar_analysis.py      ← Flags pixels exceeding threshold
│   │   └── insar_image_processing.py ← SAR image → deformation mask + overlay
│   │
│   ├── sensors/
│   │   └── sensor_preprocessing.py ← Feature engineering on IoT sensor data
│   │
│   ├── fusion/
│   │   └── data_fusion.py         ← Weighted merge of GNSS + InSAR
│   │
│   ├── anomaly_detection/
│   │   └── detect_anomalies.py    ← Isolation Forest anomaly detection
│   │
│   ├── bridges/
│   │   └── kaggle_bridge_pipeline.py ← Full ML training + bridge-level inference
│   │
│   └── visualization/
│       └── dashboard.py           ← Streamlit web dashboard
│
├── data/
│   ├── gnss/
│   │   ├── raw/gnss_raw.csv
│   │   └── processed/gnss_displacement.csv, gnss_analysis.csv
│   ├── insar/
│   │   ├── raw/sentinel_images/   ← SAR image PNGs
│   │   └── processed/insar_timeseries.csv, insar_analysis.csv, masks/, overlays/
│   ├── sensors/
│   │   ├── raw/sensor_data.csv
│   │   └── processed/sensor_features.csv
│   ├── fused/
│   │   ├── fused_displacement.csv
│   │   └── fused_training_frame.csv
│   ├── external/
│   │   └── bridge_digital_twin_dataset.csv  ← Kaggle dataset (download manually)
│   └── bridges/
│       ├── bridge_registry.csv
│       ├── bridge_predictions.csv
│       └── <bridge_id>/           ← Per-bridge data folders
│           ├── source_dataset.csv
│           ├── gnss_raw.csv
│           ├── insar_timeseries.csv
│           ├── sensor_data.csv
│           ├── predictions.csv
│           ├── insar_mask_metadata.csv
│           ├── insar_images/
│           ├── insar_masks/
│           └── insar_overlays/
│
├── models/
│   ├── sensor_anomaly_model.pkl   ← Isolation Forest model artifact
│   ├── bridge_anomaly_model.pkl   ← Stacked ensemble model artifact
│   ├── bridge_anomaly_metrics.json
│   └── bridge_xai_summary.json
│
├── docs/                          ← Additional documentation assets
└── notebooks/                     ← Jupyter notebooks (if any)
```

---

## 4. Configuration

**File:** `src/utils/config.py`

This is the **single source of truth** for all paths, thresholds, and hyperparameters. Every other module imports from here — no hardcoded paths anywhere else.

### Path Constants

| Constant | Path | Purpose |
|---|---|---|
| `PROJECT_ROOT` | Dynamically resolved | Root of the project |
| `GNSS_RAW_PATH` | `data/gnss/raw/gnss_raw.csv` | Raw GNSS measurements input |
| `GNSS_PROCESSED_PATH` | `data/gnss/processed/gnss_displacement.csv` | Displacement vectors output |
| `GNSS_ANALYSIS_PATH` | `data/gnss/processed/gnss_analysis.csv` | Analysis with threshold flags |
| `INSAR_RAW_IMAGE_DIR` | `data/insar/raw/sentinel_images/` | Folder of SAR image PNGs |
| `INSAR_PROCESSED_PATH` | `data/insar/processed/insar_timeseries.csv` | Normalized LOS timeseries |
| `INSAR_ANALYSIS_PATH` | `data/insar/processed/insar_analysis.csv` | Analysis with threshold flags |
| `INSAR_MASK_DIR` | `data/insar/processed/masks/` | Binary deformation masks |
| `INSAR_OVERLAY_DIR` | `data/insar/processed/overlays/` | Red-on-grey overlay images |
| `SENSOR_RAW_PATH` | `data/sensors/raw/sensor_data.csv` | Raw IoT sensor readings |
| `SENSOR_PROCESSED_PATH` | `data/sensors/processed/sensor_features.csv` | Engineered sensor features |
| `FUSED_OUTPUT_PATH` | `data/fused/fused_displacement.csv` | Fused GNSS+InSAR output |
| `KAGGLE_BRIDGE_DATASET_PATH` | `data/external/bridge_digital_twin_dataset.csv` | Downloaded Kaggle CSV |
| `BRIDGE_MODEL_PATH` | `models/bridge_anomaly_model.pkl` | Trained ensemble pickle |
| `BRIDGE_MODEL_METRICS_PATH` | `models/bridge_anomaly_metrics.json` | Precision/recall/F1/AUC |
| `BRIDGE_XAI_SUMMARY_PATH` | `models/bridge_xai_summary.json` | Feature importance summary |

### Domain Thresholds

| Constant | Value | Meaning |
|---|---|---|
| `GNSS_THRESHOLD_MM` | `5` mm | Vertical displacement above this is flagged |
| `INSAR_THRESHOLD_MM` | `8` mm | LOS displacement above this is flagged |
| `ANOMALY_SCORE_THRESHOLD` | `0.7` | Isolation Forest normalized score cutoff |
| `INSAR_IMAGE_MASK_THRESHOLD_QUANTILE` | `0.97` | Top 3% of deformation pixels become mask |

### ML Hyperparameters

| Constant | Value | Meaning |
|---|---|---|
| `FUSION_WEIGHT_GNSS` | `0.6` | GNSS gets 60% weight in the fused signal |
| `FUSION_WEIGHT_INSAR` | `0.4` | InSAR gets 40% weight |
| `ISOLATION_FOREST_CONTAMINATION` | `0.1` | Expects ~10% of data to be anomalous |
| `RANDOM_SEED` | `42` | Ensures reproducibility across all random operations |

---

## 5. Synthetic Data Generators

These three scripts create realistic synthetic data for testing the pipeline without a real sensor network. They all inject **known damage events** at a defined time point so you can verify the anomaly detection catches them.

### 5.1 GNSS Data Generator

**File:** `src/utils/generate_synthetic_gnss.py`
**Function:** `generate_synthetic_gnss(output_path, periods=60, seed=42)`

Generates 60 days of daily GNSS readings simulating a bridge with gradual subsidence:

- **X, Y** coordinates: Random walk with very small noise (σ = 0.002 m) → simulates lateral bedding shifts
- **Z** coordinate: Random walk with small noise (σ = 0.001 m) → vertical elevation
- **Damage injection:** Starting at day 40 (67% of the time series), Z is gradually depressed by up to **10 mm** using a linear ramp (`np.linspace(0, 0.010, ...)`), simulating progressive settlement or sinking.

**Output CSV columns:** `timestamp, x, y, z`

### 5.2 InSAR Data Generator

**File:** `src/utils/generate_synthetic_insar.py`
**Function:** `generate_synthetic_insar(output_path, image_dir, periods=30, image_size=128, seed=43)`

Generates 30 InSAR acquisitions every 12 days (Sentinel-1 revisit cadence):

- **Timeseries:** Cumulative LOS displacement starting from a random walk plus a **damage ramp** injected at period 15 (50% of the time), growing to +6 mm.
- **SAR images:** For each acquisition, a 128×128 grayscale image is generated:
  - Background: Gaussian noise (σ = 0.03) simulating radar speckle
  - Deformation signal: A Gaussian "bump" (σ = 0.23 normalized units) centred at a position that rotates around the image over time (using `sin/cos` of the acquisition date's day-of-year)
  - Intensity of the bump scales with the current LOS displacement value
  - Normalized to [0, 1] and saved as PNG

**Output:**
- `data/insar/processed/insar_timeseries.csv` — columns: `timestamp, los_displacement`
- `data/insar/raw/sentinel_images/*.png` — 30 SAR image files named `sar_YYYYMMDD.png`

### 5.3 Sensor Data Generator

**File:** `src/utils/generate_synthetic_sensor_data.py`
**Function:** `generate_synthetic_sensor_data(output_path, periods=120, seed=45)`

Generates 120 readings at 12-hour intervals (60 days) across 7 sensor channels:

| Column | Description | Pattern |
|---|---|---|
| `temperature_c` | Air temperature | Sinusoidal oscillation ± 4 °C around 28 °C + noise |
| `humidity_pct` | Relative humidity | Sinusoidal oscillation ± 9% around 58% + noise, clipped to [20, 95] |
| `vibration_rms` | Vibration RMS (g or m/s²) | Absolute normal random walk (always positive) |
| `strain_ue` | Strain in micro-strain (μϵ) | Cumulative random walk |
| `tilt_x_deg` | Tilt around X axis | Cumulative random walk (small σ = 0.005°/step) |
| `tilt_y_deg` | Tilt around Y axis | Cumulative random walk (smaller σ = 0.004°/step) |
| `acoustic_db` | Acoustic emission | Mean 37 dB + noise |

**Damage injection at reading 84 (70% mark):**
- Vibration gradually increases by +0.12 (over the remaining period)
- Strain increases by +24 μϵ
- Tilt X increases by +0.8°, Tilt Y decreases by -0.6°
- Acoustic increases by +7 dB

---

## 6. GNSS Module

### 6.1 Preprocessing

**File:** `src/gnss/gnss_preprocessing.py`

**Purpose:** Convert raw GNSS 3D coordinates into relative displacement vectors.

**Steps:**
1. **Load** `gnss_raw.csv`, validate it has `timestamp, x, y, z` columns
2. **Sort** by timestamp
3. **Compute displacement** relative to the very first reading:
   - `dx = x - x[0]`
   - `dy = y - y[0]`
   - `dz = z - z[0]`
4. **Save** `gnss_displacement.csv` with columns: `timestamp, dx, dy, dz`

**Why this matters:** Absolute coordinates (like ECEF) are meaningless for structural analysis. What matters is how much the bridge has moved from its reference position in each direction.

### 6.2 Analysis

**File:** `src/gnss/gnss_analysis.py`

**Purpose:** Convert displacement vectors into engineering-meaningful measurements and flag exceedances.

**Steps:**
1. **Load** processed displacement data
2. **Compute magnitudes** (in millimetres):
   - `horizontal_mm = sqrt(dx² + dy²) × 1000`
   - `vertical_mm = dz × 1000`
   - `total_mm = sqrt(dx² + dy² + dz²) × 1000`
3. **Flag threshold exceedance:** `threshold_exceeded = 1` if `|vertical_mm| ≥ 5 mm`

**Output columns:** `timestamp, horizontal_mm, vertical_mm, total_mm, threshold_exceeded`

The vertical component is specifically monitored because settlement/subsidence in bridges typically manifests as downward Z movement.

---

## 7. InSAR Module

### 7.1 Preprocessing

**File:** `src/insar/insar_preprocessing.py`

**Purpose:** Normalize the LOS (Line-of-Sight) displacement timeseries to a baseline-relative signal.

**Steps:**
1. **Load** `insar_timeseries.csv`, validate `timestamp` and `los_displacement` (or `los_disp_norm`) columns
2. **Normalize** relative to the first observation:
   - `los_disp_norm = los_displacement - los_displacement[0]`
3. **Save** to the same `insar_timeseries.csv`

**What is LOS displacement?** In SAR interferometry, the satellite measures the change in distance between the satellite and the ground along its line of sight. Positive LOS means the ground moved toward the satellite; negative means it moved away.

### 7.2 Analysis

**File:** `src/insar/insar_analysis.py`

**Purpose:** Flag time points where deformation exceeds the threshold.

**Steps:**
1. **Load** processed timeseries
2. **Compute absolute LOS:** `abs_los_disp = |los_disp_norm|`
3. **Flag:** `threshold_exceeded = 1` if `abs_los_disp ≥ 8 mm`

**Output columns:** `timestamp, los_disp_norm, abs_los_disp, threshold_exceeded`

### 7.3 Image Processing

**File:** `src/insar/insar_image_processing.py`

**Purpose:** Process real SAR image frames into deformation masks and visual overlays.

**Algorithm:**
1. **Load all SAR images** from the `sentinel_images/` folder (PNG/JPG/TIF)
2. **Convert to grayscale** — if the image is RGB, average the three channels
3. **Normalize** each image to [0, 1] range
4. **Use the first image as the baseline** (assumed no deformation at time 0)
5. For each subsequent image:
   - **Compute deformation:** `deformation = |current - baseline|`
   - **Threshold the mask:** Take only the top 3% of positive deformation values (`quantile = 0.97`) → pixels above this are marked as 1 (deformed), rest as 0
   - **Save mask** as greyscale PNG (white = deformed)
   - **Create overlay:** Copy of the grayscale image, with deformed pixels coloured red `[1.0, 0.1, 0.1]`
6. **Record metadata** per image: timestamp, paths to image/mask/overlay, mask ratio (fraction of pixels deformed)

---

## 8. Sensor Module

**File:** `src/sensors/sensor_preprocessing.py`

**Purpose:** Clean and feature-engineer the raw IoT sensor data.

**Steps:**
1. **Load** `sensor_data.csv`, validate `timestamp` column
2. **Convert all non-timestamp columns to numeric** (coerce any non-numeric to NaN)
3. **Fill missing values:** Linear interpolation first, then median fill for any remaining NaN
4. **Feature engineering** — derived features added:
   - `vibration_rms_roll3`: 3-reading rolling mean of vibration (smoothed trend)
   - `strain_gradient`: First difference of strain (rate of crack propagation)
   - `acoustic_roll3`: 3-reading rolling mean of acoustic emissions

These derived features help the anomaly model detect emerging patterns that are not obvious from a single point reading.

---

## 9. Data Fusion Module

**File:** `src/fusion/data_fusion.py`

**Purpose:** Merge GNSS and InSAR displacement signals into a single fused displacement estimate.

**Algorithm:**
1. **Load** processed GNSS displacement (`dz` column) and normalized InSAR timeseries (`los_disp_norm`)
2. **Time-align** using `pandas.merge_asof` with `direction='nearest'` — for each GNSS timestamp, finds the nearest InSAR observation in time. This handles the different sampling frequencies (GNSS is daily, InSAR is every 12 days).
3. **Convert GNSS to mm:** `gnss_dz_mm = dz × 1000`
4. **Compute weighted fusion:**
   ```
   fused_disp = 0.6 × gnss_dz_mm + 0.4 × los_disp_norm
   ```
   GNSS gets higher weight (60%) because it is more temporally dense and directly measures 3D position. InSAR contributes 40% as it has better spatial resolution.
5. **Save** `fused_displacement.csv`: `timestamp, gnss_dz_mm, los_disp_norm, fused_disp`

---

## 10. Anomaly Detection Module

**File:** `src/anomaly_detection/detect_anomalies.py`

**Purpose:** Detect structural anomalies using unsupervised machine learning on the fused multi-sensor dataset.

### Training Frame Construction

The function `build_training_frame()` merges four data sources into a single wide feature table using `merge_asof` (nearest-time join):

| Source | Prefix | Key Columns |
|---|---|---|
| Fused displacement | (none) | `gnss_dz_mm, los_disp_norm, fused_disp` |
| Sensor features | `sensor_` | `sensor_vibration_rms, sensor_strain_ue, ...` |
| GNSS analysis | `gnss_` | `gnss_horizontal_mm, gnss_vertical_mm, gnss_threshold_exceeded, ...` |
| InSAR analysis | `insar_` | `insar_abs_los_disp, insar_threshold_exceeded, ...` |

After joining, any missing values are filled by interpolation then median.

### Isolation Forest Model

The `train_and_predict_anomalies()` function:

1. **Standardize features** with `StandardScaler` (zero mean, unit variance) — crucial because features have different scales (mm vs °C vs dB)
2. **Train Isolation Forest** with `contamination=0.10` — expects 10% of readings to be anomalous
3. **Isolation Forest algorithm:**
   - Randomly selects a feature and a split value between the min and max of that feature
   - Recursively partitions the data
   - Anomalies require fewer splits to isolate (shorter path length = more anomalous)
4. **Decision scores:** Raw scores from `decision_function()` are negated and normalized to [0, 1]:
   - Score close to 1 = very anomalous
   - Score close to 0 = normal
5. **Final label:** A point is marked anomalous (`anomaly = 1`) if:
   - The Isolation Forest labels it `-1` (outlier), **OR**
   - Its normalized anomaly score ≥ 0.7 (the `ANOMALY_SCORE_THRESHOLD`)
6. **Model saved** as pickle: contains the `IsolationForest`, `StandardScaler`, feature column names, and score normalization bounds.

---

## 11. Bridge (Kaggle) Pipeline

**File:** `src/bridges/kaggle_bridge_pipeline.py`

This is the most complex and ML-intensive module. It trains a production-grade ensemble classifier on real bridge health monitoring data and generates per-bridge digital twins.

### Bridge Catalog

Six virtual bridges are defined in `BRIDGE_CATALOG`, each with:
- `bridge_id`, `bridge_name`, `lat`, `lon`, `city`, `region`

These represent real US cities: San Francisco, Seattle, New York, Chicago, Houston, Miami.

### Step-by-Step Pipeline (`run_kaggle_bridge_pipeline()`)

#### Step 1 — Load Dataset
Reads `bridge_digital_twin_dataset.csv` from Kaggle. Expects a `Timestamp` column. Renames it to `timestamp`, parses as datetime, drops nulls, and sorts chronologically.

#### Step 2 — Assign Bridge Instances
Each row in the dataset is assigned a `bridge_id` in a round-robin fashion across the 6 bridges. Bridge metadata (lat, lon, city, region) is merged in from the catalog.

#### Step 3 — Export Bridge Views
For each of the 6 bridges:
- Creates `data/bridges/<bridge_id>/` folder
- Derives three modality files from the bridge's rows:
  - **GNSS:** `gnss_raw.csv` — built from `Displacement_mm`, `Soil_Settlement_mm`, `Deflection_mm`
  - **InSAR timeseries:** `insar_timeseries.csv` — LOS = `Deflection_mm + 0.3 × Tilt_deg`
  - **Sensor data:** `sensor_data.csv` — all remaining columns
- Generates **72 synthetic SAR images** per bridge using a Gaussian deformation field (same algorithm as the synthetic generator), plus masks, overlays, interferograms, heatmaps, and coherence maps

#### Step 4 — Feature Engineering (`engineer_bridge_features()`)

For each of 16 core signal columns (strain, deflection, vibration, tilt, displacement, crack propagation, etc.), the following features are computed **per bridge** using `groupby("bridge_id")`:

| Feature Type | Suffix | Description |
|---|---|---|
| First difference | `_diff_1` | Rate of change between consecutive readings |
| 6-step rolling mean | `_roll_mean_6` | Short-term trend (smoothed signal) |
| 12-step rolling std | `_roll_std_12` | Short-term variability (volatility) |
| 12-step EWM | `_ewm_12` | Exponentially weighted mean (decays older readings) |

Additional interaction features:
- `deflection_displacement_ratio` = Deflection / |Displacement + ε|
- `vibration_strain_coupling` = Vibration × Strain (detects coupled failures)
- `failure_health_gap` = Probability_of_Failure - Structural_Health_Index

Temporal features: `hour`, `dayofweek`, `month`, `dayofyear`, `hour_sin/cos`, `day_sin/cos`, `bridge_age_index`

#### Step 5 — Target Construction (`_build_target()`)

A binary anomaly label is created — a row is **anomalous (1)** if ANY of these conditions hold:
- `Maintenance_Alert > 0`
- `Anomaly_Detection_Score ≥ 0.8`
- `Probability_of_Failure_PoF ≥ 0.12`
- Both `Flood_Event_Flag > 0` AND `High_Winds_Storms > 0`

#### Step 6 — Model Training

**Architecture: Stacked Ensemble (StackingClassifier)**

The model uses three base classifiers and one meta-learner:

| Model | Role | Key Hyperparameters |
|---|---|---|
| `HistGradientBoostingClassifier` | Base estimator 1 | lr=0.045, depth=6, 350 iterations |
| `ExtraTreesClassifier` | Base estimator 2 | 450 trees, balanced class weights |
| `RandomForestClassifier` | Base estimator 3 | 350 trees, balanced class weights |
| `LogisticRegression` | Meta-learner | Max iter=2500, balanced weights, liblinear solver |

**Preprocessing pipeline:**
- Numeric columns → `SimpleImputer(median)` → passed directly
- Categorical columns → `SimpleImputer(most_frequent)` → `OneHotEncoder`

**Training strategy:**
1. Split 85:15 train/test (stratified)
2. Further split training into 82.35:17.65 train/validation
3. Train base models on `train` split with **sample weights** (`compute_sample_weight("balanced")`) to handle class imbalance
4. **Optimal threshold selection** on validation set: Find the classification threshold that maximizes F1 on the precision-recall curve
5. Retrain final model on the full train+val set (`x_train_val`)
6. Evaluate on held-out test set

#### Step 7 — Evaluation Metrics

Saved to `models/bridge_anomaly_metrics.json`:
- `precision`, `recall`, `f1` (at optimal threshold)
- `average_precision` (PR-AUC)
- `roc_auc` (if both classes present in test set)
- `threshold` value used
- `train/validation/test positive rate`
- `xai_top_features` — global feature importance

#### Step 8 — Explainability (XAI)

**Global XAI:** `permutation_importance()` is computed on the test set using `average_precision` as the metric. The top 16 most important features are stored.

**Local XAI per bridge:** For each bridge, the highest-probability anomaly row is explained using **counterfactual feature ablation**:
- For each important feature, replace its value with the median reference value
- Measure how much the anomaly probability changes
- The feature with the largest probability drop is the most responsible driver
- Saved to `data/bridges/<bridge_id>/xai_top_factors.csv`

#### Step 9 — Return Value

The function returns:
```python
{
    "rows": int,           # Total dataset rows
    "bridges": int,        # Number of unique bridges
    "anomalies": int,      # Total anomalous predictions
    "threshold": float,    # Optimal decision threshold
    "metrics": dict,       # Full metrics dict
}
```

---

## 12. Visualization Dashboard

**File:** `src/visualization/dashboard.py`

A **Streamlit** web application with the following interactive features:

| Feature | Description |
|---|---|
| **Map view** | Interactive Plotly map showing all 6 bridge locations with colour-coded anomaly risk |
| **Bridge selector** | Click on map marker or use dropdown to select a bridge |
| **GNSS panel** | Time series plot of X, Y, Z displacement for the selected bridge |
| **InSAR panel** | LOS displacement timeseries for the selected bridge |
| **Sensor streams** | Multi-channel sensor data plots |
| **3D bridge digital twin** | 3D visualization with anomaly markers rendered on the bridge structure |
| **InSAR image panel** | Shows original SAR image, predicted deformation mask, and red overlay |
| **Anomaly table** | Live table of all detected anomalous events for the bridge |
| **XAI panel** | Bar chart of top anomaly drivers (feature importance) per bridge |

**Run with:**
```bash
streamlit run src/visualization/dashboard.py
```
Then open `http://localhost:8501` in your browser.

---

## 13. Entry Point — `main.py`

The main entry point orchestrates the complete pipeline in sequence:

```python
run_all_pipelines(
    generate_synthetic_data=True,
    run_kaggle=True
)
```

### Execution Order

```
1. generate_synthetic_gnss()       → data/gnss/raw/gnss_raw.csv
2. generate_synthetic_insar()      → data/insar/raw/... + insar_timeseries.csv
3. generate_synthetic_sensor_data() → data/sensors/raw/sensor_data.csv

4. run_gnss_preprocessing()        → data/gnss/processed/gnss_displacement.csv
5. run_insar_preprocessing()       → data/insar/processed/insar_timeseries.csv
6. run_sensor_preprocessing()      → data/sensors/processed/sensor_features.csv

7. run_gnss_analysis()             → data/gnss/processed/gnss_analysis.csv
8. run_insar_analysis()            → data/insar/processed/insar_analysis.csv

9. run_data_fusion()               → data/fused/fused_displacement.csv
10. run_anomaly_detection()        → fused_displacement.csv (+ anomaly cols) + model .pkl
11. process_insar_images()         → masks/, overlays/, insar_mask_metadata.csv

12. run_kaggle_bridge_pipeline()   → models/, data/bridges/
```

### CLI Arguments

```
python main.py                        # Full pipeline
python main.py --skip-generate        # Skip steps 1-3 (use existing data)
python main.py --skip-kaggle          # Skip step 12 (no Kaggle model)
python main.py --skip-generate --skip-kaggle  # Core analysis only
```

---

## 14. Data Flow Diagram

```
[RAW DATA]
gnss_raw.csv ──────────► gnss_preprocessing ──► gnss_displacement.csv
                                                         │
insar_timeseries.csv ──► insar_preprocessing ──► insar_timeseries.csv (normalized)
                                                         │
sentinel_images/*.png ──► insar_image_processing ──► masks/ + overlays/
                                                         │
sensor_data.csv ────────► sensor_preprocessing ──► sensor_features.csv

                     ┌───────────────────────────────────┤
                     │                                   │
                     ▼                                   ▼
              gnss_analysis.csv                    insar_analysis.csv
              (threshold flags)                   (threshold flags)
                     │                                   │
                     └──────────────┬────────────────────┘
                                    │
                             data_fusion
                                    │
                         fused_displacement.csv
                                    │
                          anomaly_detection
                          (Isolation Forest)
                                    │
                    fused_displacement.csv + anomaly cols
                    + sensor_anomaly_model.pkl

[KAGGLE DATA]
bridge_digital_twin_dataset.csv ──► kaggle_bridge_pipeline
                                    ├─► data/bridges/<id>/ (per-bridge views)
                                    ├─► bridge_anomaly_model.pkl
                                    ├─► bridge_anomaly_metrics.json
                                    └─► bridge_xai_summary.json

[DASHBOARD]
All outputs ──► streamlit dashboard ──► http://localhost:8501
```

---

## 15. Key Thresholds and Parameters

| Parameter | Value | Rationale |
|---|---|---|
| GNSS threshold | 5 mm vertical | Below 5 mm is within normal thermal expansion range for bridges |
| InSAR threshold | 8 mm LOS | SAR measurement noise floor is ~2–3 mm; 8 mm indicates structural movement beyond noise |
| Isolation Forest contamination | 10% | Standard initial assumption for infrastructure anomaly detection |
| Anomaly score cutoff | 0.70 | Balances sensitivity and false alarm rate |
| Fusion GNSS weight | 0.60 | Higher because GNSS is daily (6× more frequent than InSAR) |
| Fusion InSAR weight | 0.40 | Lower frequency but higher spatial resolution |
| InSAR mask quantile | 0.97 | Only the top 3% most deformed pixels are flagged to minimize false positives |
| Ensemble contamination | `class_weight="balanced"` | Handles typical 10–20% positive rate in bridge anomaly datasets |

---

## 16. Generated Outputs

After a full `python main.py` run, the following files are created:

### Global (Synthetic Pipeline)
| File | Description |
|---|---|
| `data/gnss/raw/gnss_raw.csv` | 60-day GNSS position data |
| `data/gnss/processed/gnss_displacement.csv` | dx, dy, dz relative to day 0 |
| `data/gnss/processed/gnss_analysis.csv` | Horizontal/vertical/total mm + flag |
| `data/insar/raw/sentinel_images/` | 30 SAR PNGs |
| `data/insar/processed/insar_timeseries.csv` | Normalized LOS timeseries |
| `data/insar/processed/insar_analysis.csv` | LOS + threshold flag |
| `data/insar/processed/masks/` | Binary deformation mask PNGs |
| `data/insar/processed/overlays/` | Red-highlighted deformation overlays |
| `data/insar/processed/insar_mask_metadata.csv` | Paths + mask ratios per image |
| `data/sensors/raw/sensor_data.csv` | 120 half-daily sensor readings |
| `data/sensors/processed/sensor_features.csv` | Cleaned + feature-engineered sensors |
| `data/fused/fused_displacement.csv` | GNSS+InSAR fused signal + anomaly labels |
| `data/fused/fused_training_frame.csv` | Full wide training frame |
| `models/sensor_anomaly_model.pkl` | Trained Isolation Forest artifact |

### Per-Bridge (Kaggle Pipeline)
| File | Description |
|---|---|
| `data/bridges/bridge_registry.csv` | All 6 bridge metadata |
| `data/bridges/bridge_predictions.csv` | Predictions for all rows |
| `data/bridges/<id>/source_dataset.csv` | Bridge's rows from Kaggle dataset |
| `data/bridges/<id>/gnss_raw.csv` | Derived GNSS for this bridge |
| `data/bridges/<id>/insar_timeseries.csv` | Derived InSAR for this bridge |
| `data/bridges/<id>/sensor_data.csv` | Sensor features for this bridge |
| `data/bridges/<id>/predictions.csv` | Anomaly predictions per row |
| `data/bridges/<id>/insar_mask_metadata.csv` | 72 InSAR frame metadata |
| `data/bridges/<id>/insar_images/*.png` | 72 SAR images |
| `data/bridges/<id>/insar_masks/*.png` | 72 deformation masks |
| `data/bridges/<id>/insar_overlays/*.png` | 72 overlay images |
| `data/bridges/<id>/xai_top_factors.csv` | Local XAI for top anomaly row |
| `models/bridge_anomaly_model.pkl` | Trained stacked ensemble |
| `models/bridge_anomaly_metrics.json` | Precision, recall, F1, AUC |
| `models/bridge_xai_summary.json` | Global feature importances |

---

## 17. Setup and Running the Project

### Prerequisites
- Python 3.11 (recommended; 3.10 also works)
- PowerShell (Windows) or Bash (Linux/macOS)
- A Kaggle account (for downloading the dataset)

### Step 1 — Install Dependencies

```powershell
cd "c:\Users\Vijayakrishnaji\Desktop\capstone mainproject\Structural-Health-Monitoring"
pip install -r requirements.txt
```

### Step 2 — Set Kaggle API Token

Get your token from `kaggle.com → Settings → API → Create New Token`.

```powershell
# For the current session only:
$env:KAGGLE_API_TOKEN="your_kaggle_api_key_here"

# To make it permanent across all sessions:
[System.Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", "your_kaggle_api_key_here", "User")
```

### Step 3 — Download Kaggle Dataset

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\Scripts\kaggle.exe" datasets download `
  -d mithil27360/digital-twin-bridge-structural-health-monitoring `
  -p data/external --unzip
```

The downloaded file should appear at `data/external/bridge_digital_twin_dataset.csv`.

### Step 4 — Run the Pipeline

```powershell
# Full run (recommended first time):
python main.py

# Skip synthetic data generation (reuse existing CSVs):
python main.py --skip-generate

# Skip Kaggle bridge model (no dataset required):
python main.py --skip-kaggle

# Fastest option (only core analysis, no Kaggle, reuse data):
python main.py --skip-generate --skip-kaggle
```

### Step 5 — Launch the Dashboard

```powershell
streamlit run src/visualization/dashboard.py
```

Open your browser at **http://localhost:8501**.

---

## 18. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | Latest | Numerical arrays, random number generation, matrix math |
| `pandas` | Latest | DataFrames, CSV I/O, time-series operations, merge_asof |
| `matplotlib` | Latest | Saving SAR images and masks as PNG files |
| `scikit-learn` | Latest | IsolationForest, StackingClassifier, StandardScaler, metrics |
| `plotly` | Latest | Interactive maps and charts in the dashboard |
| `streamlit` | Latest | Web dashboard framework |
| `kaggle` | Latest | Downloading datasets from Kaggle via CLI |

All are installable via: `pip install -r requirements.txt`

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **GNSS** | Global Navigation Satellite System — umbrella term for GPS, GLONASS, Galileo, BeiDou |
| **InSAR** | Interferometric Synthetic Aperture Radar — measures surface deformation using satellite radar phase differences |
| **LOS** | Line-of-Sight — the direction from the satellite to the ground point |
| **Displacement** | Physical movement of the structure from a reference position |
| **dx, dy, dz** | Relative displacement in East, North, and Vertical directions respectively |
| **Rolling mean/std** | Statistics computed over a sliding window of the most recent N readings |
| **EWM** | Exponentially Weighted Mean — like a rolling mean but recent values count more |
| **Isolation Forest** | Unsupervised anomaly detection algorithm that isolates outliers via random partitioning |
| **Stacking Classifier** | Ensemble method where base model predictions are used as inputs to a meta-model |
| **Precision** | Fraction of flagged anomalies that are truly anomalous (TP / (TP + FP)) |
| **Recall** | Fraction of true anomalies that were caught (TP / (TP + FN)) |
| **F1 Score** | Harmonic mean of precision and recall |
| **PR-AUC** | Area under the Precision-Recall curve — better than ROC-AUC for imbalanced datasets |
| **XAI** | eXplainable AI — techniques to interpret ML predictions |
| **Contamination** | In Isolation Forest: the expected fraction of anomalies in the training data |
| **merge_asof** | Pandas time-series join that aligns rows to the nearest timestamp |
| **Digital Twin** | A virtual replica of a physical structure, updated with real sensor data |
| **SAR** | Synthetic Aperture Radar — a radar imaging technique used by satellites like Sentinel-1 |
| **Speckle** | Grainy noise in SAR images caused by coherent scattering of radar waves |
| **μϵ (micro-strain)** | Unit of strain measurement: one part per million deformation |

---

*Documentation generated for the Structural Health Monitoring — Bridge Digital Twin project.*
*Last updated: March 2026*
