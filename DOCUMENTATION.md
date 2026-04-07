# Structural Health Monitoring (SHM) — Project Documentation

> **Capstone Project** | Multi-modal bridge health monitoring system powered by machine learning, satellite data, and AI agents.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Getting Started](#4-getting-started)
5. [Core Pipeline Modules](#5-core-pipeline-modules)
   - [GNSS Module](#51-gnss-module)
   - [InSAR Module](#52-insar-module)
   - [Sensor Module](#53-sensor-module)
   - [Data Fusion Module](#54-data-fusion-module)
   - [Anomaly Detection Module](#55-anomaly-detection-module)
   - [Bridge Pipeline Module](#56-bridge-pipeline-module)
6. [REST API](#6-rest-api)
7. [CrewAI Agent System](#7-crewai-agent-system)
8. [Frontend Dashboard](#8-frontend-dashboard)
9. [Data Model](#9-data-model)
10. [Configuration Reference](#10-configuration-reference)
11. [CLI Reference](#11-cli-reference)
12. [Environment Variables](#12-environment-variables)
13. [Key ML Models](#13-key-ml-models)

---

## 1. Project Overview

This project is a full-stack **Structural Health Monitoring (SHM)** platform designed to detect, analyze, and report anomalies in bridge infrastructure using multiple sensing modalities and machine learning.

### What It Does

| Capability | Description |
|---|---|
| **Multi-modal data ingestion** | Processes GNSS (GPS displacement), InSAR (satellite radar), and IoT sensor data |
| **Anomaly detection** | Trains and runs an Isolation Forest model on fused sensor data |
| **Bridge digital twin** | Maintains a per-bridge dataset (telemetry, predictions, images) for 6 monitored bridges |
| **ML ensemble model** | Trains a stacked classifier (HistGBM + ExtraTrees + RandomForest → LogisticRegression) to predict bridge anomalies from Kaggle data |
| **Explainability (XAI)** | Computes counterfactual feature impacts to explain model predictions |
| **CrewAI agents** | Deploys 4 specialized LLM-powered agents to synthesize telemetry into executive reports |
| **REST API** | FastAPI backend exposing all pipeline results as structured JSON |
| **React dashboard** | Vite/TypeScript frontend for interactive fleet-level and bridge-level analysis |

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Data Sources                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────────┐   │
│  │   GNSS   │  │  InSAR   │  │  Sensors  │  │  Kaggle Bridge   │   │
│  │ (GPS XYZ)│  │(SAR imgs)│  │(IoT data) │  │  Digital Twin    │   │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  └────────┬─────────┘   │
└───────┼─────────────┼──────────────┼───────────────────┼────────────┘
        │             │              │                   │
┌───────▼─────────────▼──────────────▼───────────────────▼────────────┐
│                     Preprocessing Layer                               │
│   gnss_preprocessing → insar_preprocessing → sensor_preprocessing    │
└───────────────────────────┬──────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│                     Analysis & Fusion Layer                            │
│         gnss_analysis + insar_analysis → data_fusion                  │
│         kaggle_bridge_pipeline (train + predict)                       │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│                     Anomaly Detection Layer                            │
│         detect_anomalies (Isolation Forest on fused data)             │
│         bridge_anomaly_model (Stacked Ensemble on bridge telemetry)    │
└────────────────┬────────────────────────────┬────────────────────────┘
                 │                            │
    ┌────────────▼──────────┐    ┌────────────▼──────────────┐
    │   FastAPI REST API     │    │     CrewAI Agent Crew      │
    │  /api/overview         │    │  Fleet Intelligence Analyst│
    │  /api/bridges/:id      │    │  Structural Triage Engineer│
    │  /api/reports          │    │  Dashboard Strategist      │
    │  /api/pipeline/run     │    │  Executive Reporting Lead  │
    └────────────┬──────────┘    └────────────┬──────────────┘
                 │                            │
    ┌────────────▼────────────────────────────▼──────────────┐
    │              React / Vite Frontend Dashboard             │
    │  Fleet Overview · Bridge Detail · InSAR Viewer          │
    │  XAI Explainability · Reports Workspace                 │
    └────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
Capstone_Vijay/
│
├── main.py                    # CLI entry point — runs the full pipeline
├── requirements.txt           # Python dependencies
├── .env / .env.example        # Environment variables (API keys, LLM config)
│
├── src/                       # All Python source code
│   ├── __init__.py
│   ├── agents/                # CrewAI multi-agent system
│   │   ├── crew.py            # Agent definitions, tasks, crew orchestration
│   │   ├── tools.py           # Agent tool implementations (read fleet, write report, etc.)
│   │   └── prompts.py         # Optional prompt templates
│   │
│   ├── api/                   # FastAPI REST backend
│   │   ├── main.py            # Route definitions
│   │   └── services.py        # Business logic layer (data access, computations)
│   │
│   ├── anomaly_detection/
│   │   └── detect_anomalies.py  # Isolation Forest anomaly detection
│   │
│   ├── bridges/
│   │   └── kaggle_bridge_pipeline.py  # Dataset loading, training, inference, XAI
│   │
│   ├── fusion/
│   │   └── data_fusion.py     # GNSS + InSAR weighted fusion
│   │
│   ├── gnss/
│   │   ├── gnss_preprocessing.py  # Raw GNSS → displacement
│   │   └── gnss_analysis.py       # Threshold detection, smooth signals
│   │
│   ├── insar/
│   │   ├── insar_preprocessing.py    # Raw InSAR → normalized time-series
│   │   ├── insar_analysis.py         # Threshold detection
│   │   └── insar_image_processing.py # SAR image mask generation
│   │
│   ├── sensors/
│   │   └── sensor_preprocessing.py  # IoT sensor feature extraction
│   │
│   ├── utils/
│   │   ├── config.py                        # All path constants and thresholds
│   │   ├── generate_synthetic_gnss.py       # Synthetic GNSS data generator
│   │   ├── generate_synthetic_insar.py      # Synthetic InSAR data generator
│   │   └── generate_synthetic_sensor_data.py # Synthetic IoT sensor generator
│   │
│   └── visualization/                       # (reserved for plot utilities)
│
├── data/                       # All data files (auto-created by pipeline)
│   ├── external/               # Raw Kaggle bridge dataset CSV
│   ├── gnss/                   # GNSS raw and processed data
│   ├── insar/                  # InSAR raw images and processed masks
│   ├── sensors/                # IoT sensor raw and processed data
│   ├── fused/                  # Merged GNSS+InSAR output
│   └── bridges/                # Per-bridge digital twin data
│       ├── bridge_registry.csv
│       ├── bridge_predictions.csv
│       └── {bridge_id}/        # Per-bridge subdirectory
│           ├── source_dataset.csv
│           ├── gnss_raw.csv
│           ├── insar_timeseries.csv
│           ├── sensor_data.csv
│           ├── predictions.csv
│           ├── xai_top_factors.csv
│           └── insar_images/, insar_masks/, ...
│
├── models/                     # Trained ML model artifacts
│   ├── sensor_anomaly_model.pkl
│   ├── bridge_anomaly_model.pkl
│   ├── bridge_anomaly_metrics.json
│   └── bridge_xai_summary.json
│
├── reports/                    # CrewAI-generated markdown + dashboard reports
│
├── frontend/                   # React/Vite/TypeScript dashboard
│   ├── src/
│   │   ├── App.tsx             # Main single-page application
│   │   ├── types.ts            # TypeScript type definitions
│   │   ├── styles.css          # Stylesheet
│   │   └── main.tsx            # Entry point
│   └── dist/                   # Production build (served by FastAPI)
│
├── docs/                       # Additional documentation
│   ├── streamlit_instructions.md
│   └── user_flow.md
│
├── notebooks/                  # Jupyter notebooks (exploratory)
└── .cache/                     # Runtime caches (matplotlib, crewai)
```

---

## 4. Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for the frontend)
- A Kaggle bridge dataset CSV at `data/external/bridge_digital_twin_dataset.csv`
- An API key for your preferred LLM provider (Gemini or OpenAI) if using agents

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Capstone_Vijay

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and fill in your API keys
```

### Run the Pipeline (CLI)

```bash
# Full pipeline with synthetic data generation + Kaggle training
python main.py

# Skip data generation (reuse existing files)
python main.py --skip-generate

# Skip Kaggle bridge training
python main.py --skip-kaggle

# Run with CrewAI agents (requires an API key)
python main.py --run-agents --llm-provider gemini
```

### Run the API Server

```bash
uvicorn src.api.main:app --reload --port 8000
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Run the Frontend (Development)

```bash
cd frontend
npm install
npm run dev
```

The dashboard will be served at `http://localhost:5173` and proxies API calls to the backend.

### Build the Frontend (Production)

```bash
cd frontend
npm run build
```

The built `dist/` folder is automatically detected and served by FastAPI at the root path.

---

## 5. Core Pipeline Modules

The pipeline runs in sequential stages. Each stage reads output from the previous one and writes to the `data/` folder.

```
Synthetic Data → GNSS Preprocessing → InSAR Preprocessing → Sensor Preprocessing
     → GNSS Analysis → InSAR Analysis
     → Data Fusion → Anomaly Detection
     → Bridge Pipeline (Train + Predict + XAI)
```

### 5.1 GNSS Module

**Files:** `src/gnss/gnss_preprocessing.py`, `src/gnss/gnss_analysis.py`

#### Preprocessing (`run_gnss_preprocessing`)
- Reads raw GNSS CSV (`data/gnss/raw/gnss_raw.csv`)
- Computes displacement deltas from a baseline reading (columns: `dx`, `dy`, `dz`)
- Writes processed displacement to `data/gnss/processed/gnss_displacement.csv`

#### Analysis (`run_gnss_analysis`)
- Reads processed displacement
- Computes:
  - `horizontal_mm` — horizontal displacement magnitude (√dx²+dy²) in millimeters
  - `vertical_mm` — vertical displacement in millimeters
  - `total_mm` — 3D Euclidean displacement magnitude
  - `*_smooth` variants — 5-point rolling mean for visualization
  - `total_mm_norm` — min-max normalized total displacement
  - `threshold_exceeded` — binary flag (1 if `|vertical_mm| ≥ GNSS_THRESHOLD_MM`)
- Writes to `data/gnss/processed/gnss_analysis.csv`

**Key threshold:** `GNSS_THRESHOLD_MM = 5` (configurable in `config.py`)

---

### 5.2 InSAR Module

**Files:** `src/insar/insar_preprocessing.py`, `src/insar/insar_analysis.py`, `src/insar/insar_image_processing.py`

#### Preprocessing (`run_insar_preprocessing`)
- Reads raw InSAR time-series data and SAR images from `data/insar/raw/`
- Normalizes Line-Of-Sight (LOS) displacement to `los_disp_norm` column
- Writes to `data/insar/processed/insar_timeseries.csv`

#### Analysis (`run_insar_analysis`)
- Applies threshold detection: `threshold_exceeded = 1` if displacement ≥ `INSAR_THRESHOLD_MM`
- Writes to `data/insar/processed/insar_analysis.csv`

**Key threshold:** `INSAR_THRESHOLD_MM = 8` (configurable in `config.py`)

#### Image Processing (`process_insar_images`)
- Reads raw SAR images from `data/insar/raw/sentinel_images/`
- For each image, generates:
  - **Deformation mask** — pixels exceeding the 97th percentile of positive deformation
  - **Interferogram** — phase-encoded difference between current and baseline frames
  - **Heatmap** — magnitude of deformation colored with magma colormap
  - **Coherence map** — signal coherence inversely proportional to deformation
  - **Overlay** — grayscale SAR image with mask pixels highlighted in red
- Writes asset paths and statistics to `data/insar/processed/insar_mask_metadata.csv`

---

### 5.3 Sensor Module

**Files:** `src/sensors/sensor_preprocessing.py`

- Reads raw IoT sensor CSV: `data/sensors/raw/sensor_data.csv`
- Selects and normalizes structural sensor feature columns such as:
  - `Strain_microstrain`, `Deflection_mm`, `Vibration_ms2`, `Tilt_deg`
  - `Temperature_C`, `Humidity_percent`
  - `Probability_of_Failure_PoF`, `Structural_Health_Index_SHI`
- Writes processed features to `data/sensors/processed/sensor_features.csv`

---

### 5.4 Data Fusion Module

**File:** `src/fusion/data_fusion.py`

Merges GNSS and InSAR data streams into a single fused displacement signal.

**Algorithm:**
1. Reads `gnss_displacement.csv` and `insar_timeseries.csv`
2. Time-aligns using `pd.merge_asof` with nearest-timestamp matching
3. Computes the weighted fusion:

```
fused_disp = 0.6 × gnss_dz_mm + 0.4 × los_disp_norm
```

**Output columns:** `timestamp`, `gnss_dz_mm`, `los_disp_norm`, `fused_disp`

| Weight | Source | Rationale |
|---|---|---|
| 0.6 | GNSS | Higher-frequency, directly measured vertical displacement |
| 0.4 | InSAR | Lower-frequency, spatially comprehensive satellite measurement |

---

### 5.5 Anomaly Detection Module

**File:** `src/anomaly_detection/detect_anomalies.py`

**Model:** Scikit-learn `IsolationForest` — an unsupervised tree-based anomaly detection algorithm.

#### Training Frame Construction (`build_training_frame`)
1. Starts with the fused displacement DataFrame
2. Time-joins (nearest-neighbor) additional data sources with column prefixes:
   - `sensor_` → IoT sensor features
   - `gnss_` → GNSS analysis outputs
   - `insar_` → InSAR analysis outputs
3. Handles missing values via interpolation and median fill

#### Anomaly Detection (`train_and_predict_anomalies`)
1. Scales all numeric features with `StandardScaler`
2. Fits `IsolationForest` with `contamination = ISOLATION_FOREST_CONTAMINATION` (default: 0.1)
3. Assigns each row an `anomaly_score` (normalized 0–1) and binary `anomaly` flag:
   - `anomaly = 1` if `IsolationForest` labels as outlier **OR** normalized score ≥ 0.7

**Outputs:**
- Updated `data/fused/fused_displacement.csv` with `anomaly_score` and `anomaly` columns
- `data/fused/fused_training_frame.csv` — the full feature matrix before scoring
- `models/sensor_anomaly_model.pkl` — serialized model + scaler + metadata

---

### 5.6 Bridge Pipeline Module

**File:** `src/bridges/kaggle_bridge_pipeline.py`

This is the most complex module. It processes an external Kaggle bridge dataset to train a supervised ensemble model and generates per-bridge digital twins.

#### Monitored Bridge Catalog

| Bridge ID | Bridge Name | Location |
|---|---|---|
| `bridge_alpha` | Pacific Crown | San Francisco, CA |
| `bridge_beta` | Sound Span | Seattle, WA |
| `bridge_gamma` | Hudson Relay | New York, NY |
| `bridge_delta` | Lakeshore Axis | Chicago, IL |
| `bridge_epsilon` | Gulf Meridian | Houston, TX |
| `bridge_zeta` | Atlantic Veil | Miami, FL |

#### Pipeline Steps (`run_kaggle_bridge_pipeline`)

1. **Load Dataset** — Reads `data/external/bridge_digital_twin_dataset.csv`
2. **Assign Bridges** — Distributes rows across the 6 bridge IDs in round-robin fashion
3. **Export Bridge Views** — For each bridge, materializes:
   - `source_dataset.csv` — full source rows for that bridge
   - `gnss_raw.csv` — derived GNSS coordinates from sensor readings
   - `insar_timeseries.csv` — derived InSAR LOS displacement
   - `sensor_data.csv` — all sensor columns
   - InSAR image assets (SAR images, masks, interferograms, heatmaps, coherence maps)
4. **Feature Engineering** (`engineer_bridge_features`) — Adds:
   - Temporal features: `hour`, `dayofweek`, `month`, cyclical encodings
   - Per-bridge rolling statistics: diff-1, rolling mean (6), rolling std (12), EWM (12)
   - Interaction features: deflection/displacement ratio, vibration×strain coupling, failure–health gap
5. **Target Construction** (`_build_target`) — Binary anomaly label is `1` if any of:
   - `Maintenance_Alert > 0`
   - `Anomaly_Detection_Score ≥ 0.8`
   - `Probability_of_Failure_PoF ≥ 0.12`
   - Both `Flood_Event_Flag > 0` and `High_Winds_Storms > 0`
6. **Model Training** (`train_bridge_anomaly_model`):
   - Splits data: 70% train, 15% val, 15% test (stratified)
   - Trains a `StackingClassifier`:
     - `HistGradientBoostingClassifier` (learning_rate=0.045, max_depth=6, max_iter=350)
     - `ExtraTreesClassifier` (n_estimators=450, balanced weights)
     - `RandomForestClassifier` (n_estimators=350, balanced weights)
     - Meta-learner: `LogisticRegression` (balanced, liblinear)
   - Calibrates the decision threshold on the validation set using F1-optimal PR-curve
   - Retrains final model on train+val combined
7. **Global XAI** — Runs permutation importance on test set (PR-AUC scoring), identifies top-16 features
8. **Per-Bridge Inference** — Runs inference for each bridge and writes `predictions.csv` and `xai_top_factors.csv`

#### Metrics Reported

| Metric | Description |
|---|---|
| `precision` | Precision at optimal threshold |
| `recall` | Recall at optimal threshold |
| `f1` | F1 score |
| `average_precision` | Area under Precision-Recall curve (PR-AUC) |
| `roc_auc` | Area under ROC curve |
| `threshold` | The calibrated decision threshold |

#### Explainability (`explain_prediction_row`)

For a given prediction row, performs **counterfactual feature ablation**:
- Replaces each feature value with the training-set median (reference value)
- Measures the change in predicted anomaly probability
- Reports: `impact`, `contribution_share`, `normalized_impact`, `probability_drop`, `saturated_counterfactual`

---

## 6. REST API

**File:** `src/api/main.py`  
**Base URL:** `http://localhost:8000`  
**Framework:** FastAPI

### Endpoints

#### `GET /api/health`
Health check.
```json
{ "status": "ok" }
```

---

#### `GET /api/overview`
Returns fleet-level summary for all 6 monitored bridges.

**Response:**
```json
{
  "fleet_metrics": {
    "total_bridges": 6,
    "bridges_with_alerts": 3,
    "peak_probability": 0.9821,
    "average_probability": 0.4723,
    "highest_risk_bridge": "Pacific Crown"
  },
  "bridges": [ /* BridgeCard objects */ ],
  "model_metrics": { /* precision, recall, f1, pr_auc, threshold, ... */ },
  "reports": [ /* ReportSummary objects */ ]
}
```

---

#### `GET /api/bridges/{bridge_id}`
Returns detailed telemetry and predictions for a single bridge.

**Path parameters:** `bridge_id` — one of the 6 bridge IDs (e.g., `bridge_alpha`)

**Response includes:**
- `bridge` — summary statistics
- `telemetry.gnss` — GNSS time-series with `x`, `y`, `z`, `total_mm`
- `telemetry.insar` — InSAR LOS displacement time-series
- `telemetry.sensors` — Sensor feature time-series
- `insar_frames` — InSAR image asset URLs and metadata
- `xai_factors` — Top feature impacts for the highest-risk prediction row
- `validation` — 5-check model validation scorecard
- `hotspots` — Anomaly hotspot zones with coordinates and hit counts
- `anomalies` — Filtered anomaly rows with key telemetry columns
- `runtime_trace` — Stage-by-stage timing from the inference run
- `report_count` — Number of CrewAI reports mentioning this bridge

---

#### `POST /api/bridges/{bridge_id}/refresh`
Triggers a background re-inference for a specific bridge.

**Response:** Operation object with `id` for polling status.

---

#### `POST /api/pipeline/run`
Triggers the full SHM pipeline in a background thread.

**Query params:** `generate_synthetic_data` (bool, default: `false`)

**Response:** Operation object with `id` for polling status.

---

#### `GET /api/operations/{operation_id}`
Poll the status of a background operation.

**Response:**
```json
{
  "id": "abc123",
  "kind": "pipeline",
  "status": "running",       // "running" | "completed" | "failed"
  "progress": 65,            // 0-100
  "steps": [                 // completed pipeline stages
    { "stage": "...", "detail": "...", "status": "completed", "timestamp": "..." }
  ],
  "result": { ... },         // populated when status = "completed"
  "error": null
}
```

---

#### `GET /api/reports`
Lists all available reports in the `reports/` directory.

```json
{
  "reports": [
    {
      "name": "crew_dashboard_20240101_120000.md",
      "title": "Fleet Health Executive Brief",
      "updated_at": "2024-01-01T12:00:00",
      "size_bytes": 8192,
      "kind": "crew_dashboard",
      "provider": "gemini",
      "model": "gemini/gemini-2.5-pro"
    }
  ]
}
```

---

#### `GET /api/reports/{name}`
Returns the full content and parsed dashboard for a specific report.

**Response includes:**
- `content` — raw Markdown text
- `dashboard` — parsed dashboard payload (hero, KPIs, priority actions, charts, etc.)
- `metadata` — generation metadata (provider, model, focus bridges)

---

#### `POST /api/reports/generate`
Triggers the CrewAI agent crew to generate a new report.

**Response:** Operation object for polling.

---

#### `GET /assets/data/{path}`
Serves static data assets (e.g., InSAR images) directly from the filesystem.

---

## 7. CrewAI Agent System

**Files:** `src/agents/crew.py`, `src/agents/tools.py`, `src/agents/prompts.py`

The system employs 4 specialized AI agents that collaborate in a **hierarchical process** to produce an executive dashboard report.

### Agents

#### 1. Fleet Intelligence Analyst
- **Role:** Synthesizes fleet-level telemetry into executive-ready intelligence
- **Tools:** `ReadFleetOverviewTool`, `ReadModelMetricsTool`, `ReadBridgeTelemetryTool`
- **Output Schema:** `ExecutiveNarrative` — headline, fleet status, summary points, watchlist, model commentary, operational notes

#### 2. Structural Triage Engineer
- **Role:** Ranks bridges by risk and prescribes intervention actions
- **Tools:** `ReadBridgePredictionsTool`, `ReadBridgeXAITool`, `ReadBridgeTelemetryTool`
- **Output Schema:** `TriageSummary` — ranked bridge priorities, action counts, consensus notes, maintenance queue

#### 3. Analytics Dashboard Strategist
- **Role:** Designs the admin dashboard narrative and visual callouts
- **Tools:** `ReadFleetOverviewTool`, `ReadBridgePredictionsTool`
- **Output Schema:** `DashboardVisualPlan` — hero message, chart annotations, panel callouts, operator prompts

#### 4. Executive Reporting Lead
- **Role:** Produces the archival engineering brief
- **Tools:** `WriteReportTool`
- **Output Schema:** `ReportPacket` — report title, executive brief, admin recommendations, full markdown report

### Agent Tools

| Tool | What It Does |
|---|---|
| `ReadFleetOverviewTool` | Calls the `/api/overview` service function |
| `ReadModelMetricsTool` | Reads `models/bridge_anomaly_metrics.json` |
| `ReadBridgePredictionsTool` | Reads `data/bridges/{id}/predictions.csv` |
| `ReadBridgeTelemetryTool` | Reads GNSS, InSAR, and sensor CSVs for a bridge |
| `ReadBridgeXAITool` | Reads `data/bridges/{id}/xai_top_factors.csv` |
| `WriteReportTool` | Saves the final markdown report to `reports/` with a companion JSON artifact |

### Supported LLM Providers

| Provider | Env Variable | Default Model |
|---|---|---|
| `gemini` (default) | `GEMINI_API_KEY` | `gemini/gemini-2.5-pro` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |

### Report Artifact Structure

Each CrewAI run produces two files in `reports/`:
- `crew_dashboard_{timestamp}.md` — Markdown report
- `crew_dashboard_{timestamp}.dashboard.json` — Companion JSON with all agent outputs, metadata, and dashboard payload

---

## 8. Frontend Dashboard

**Location:** `frontend/`  
**Stack:** React 18, TypeScript, Vite

### Key Views

| View | Description |
|---|---|
| **Fleet Overview** | Map and list of all 6 bridges with anomaly counts and risk indicators |
| **Bridge Detail** | Dedicated page per bridge with telemetry charts, InSAR frames, XAI waterfall, hotspot map |
| **Anomaly Log** | Filterable table of detected anomaly rows |
| **InSAR Viewer** | Image browser showing SAR, mask, overlay, interferogram, heatmap, coherence |
| **Reports Workspace** | List and viewer for CrewAI-generated executive reports and dashboards |
| **Pipeline Runner** | UI to trigger pipeline/refresh operations and watch live progress |

### Key TypeScript Types (`src/types.ts`)

| Type | Description |
|---|---|
| `BridgeCard` | Fleet-level summary for one bridge |
| `BridgeDetail` | Full detail including telemetry, XAI, validation, hotspots |
| `OverviewResponse` | Top-level API response for fleet overview |
| `AnomalyRow` | A single anomaly detection result row |
| `InSarFrame` | Paths and metrics for one InSAR image set |
| `XaiFactor` | One feature's counterfactual impact on a prediction |
| `ValidationSummary` | 5-check model validation scorecard |
| `Hotspot` | A bridge structural zone with anomaly statistics |
| `CrewDashboard` | Full parsed dashboard from a CrewAI report |
| `OperationStatus` | Background task progress and step log |

---

## 9. Data Model

### Bridge Registry (`data/bridges/bridge_registry.csv`)
| Column | Type | Description |
|---|---|---|
| `bridge_id` | string | Unique identifier (e.g., `bridge_alpha`) |
| `bridge_name` | string | Human-readable name |
| `lat`, `lon` | float | Geographic coordinates |
| `city`, `region` | string | Location |

### Bridge Predictions (`data/bridges/{id}/predictions.csv`)
| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Observation time |
| `bridge_id` | string | Bridge identifier |
| `anomaly_probability` | float [0,1] | Model-predicted anomaly probability |
| `anomaly` | int {0,1} | Binary anomaly flag (threshold-applied) |
| `Deflection_mm` | float | Bending deflection |
| `Displacement_mm` | float | Lateral displacement |
| `Vibration_ms2` | float | Vibration acceleration |
| `Strain_microstrain` | float | Structural strain |
| `Structural_Health_Index_SHI` | float | Composite health score |
| `Probability_of_Failure_PoF` | float | Raw failure probability |
| `Vibration_Anomaly_Location` | string | Bridge zone (Deck/Tower/Cable/Pier/Joint) |
| `Simulated_Localized_Stress_Index` | float | Localized stress metric |

### GNSS Data (`data/bridges/{id}/gnss_raw.csv`)
| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Observation time |
| `x`, `y`, `z` | float | GPS coordinate (meters) |

### InSAR Time-series (`data/bridges/{id}/insar_timeseries.csv`)
| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Observation time |
| `los_displacement` | float | Line-of-sight displacement (mm) |

### InSAR Mask Metadata (`data/bridges/{id}/insar_mask_metadata.csv`)
| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Frame acquisition time |
| `image_path` | string | Path to SAR image |
| `mask_path` | string | Path to deformation mask |
| `overlay_path` | string | Path to overlay (mask on SAR) |
| `interferogram_path` | string | Path to phase interferogram |
| `heatmap_path` | string | Path to deformation heatmap |
| `coherence_path` | string | Path to coherence map |
| `mask_ratio` | float | Fraction of masked pixels |
| `deformation_energy` | float | Mean normalized deformation |
| `coherence_mean` | float | Mean coherence value |

---

## 10. Configuration Reference

**File:** `src/utils/config.py`

### Path Constants

| Constant | Default Path |
|---|---|
| `PROJECT_ROOT` | Inferred from `config.py` location |
| `DATA_DIR` | `{root}/data` |
| `MODELS_DIR` | `{root}/models` |
| `EXTERNAL_DATA_DIR` | `{root}/data/external` |
| `BRIDGES_DIR` | `{root}/data/bridges` |
| `GNSS_RAW_PATH` | `data/gnss/raw/gnss_raw.csv` |
| `GNSS_PROCESSED_PATH` | `data/gnss/processed/gnss_displacement.csv` |
| `GNSS_ANALYSIS_PATH` | `data/gnss/processed/gnss_analysis.csv` |
| `INSAR_PROCESSED_PATH` | `data/insar/processed/insar_timeseries.csv` |
| `SENSOR_PROCESSED_PATH` | `data/sensors/processed/sensor_features.csv` |
| `FUSED_OUTPUT_PATH` | `data/fused/fused_displacement.csv` |
| `KAGGLE_BRIDGE_DATASET_PATH` | `data/external/bridge_digital_twin_dataset.csv` |
| `BRIDGE_MODEL_PATH` | `models/bridge_anomaly_model.pkl` |
| `BRIDGE_MODEL_METRICS_PATH` | `models/bridge_anomaly_metrics.json` |
| `SENSOR_ANOMALY_MODEL_PATH` | `models/sensor_anomaly_model.pkl` |

### Thresholds and Hyperparameters

| Constant | Default | Description |
|---|---|---|
| `GNSS_THRESHOLD_MM` | `5` | Vertical displacement (mm) triggering a GNSS alert |
| `INSAR_THRESHOLD_MM` | `8` | LOS displacement (mm) triggering an InSAR alert |
| `ANOMALY_SCORE_THRESHOLD` | `0.7` | Normalized Isolation Forest score for anomaly labeling |
| `FUSION_WEIGHT_GNSS` | `0.6` | Weight of GNSS signal in fused displacement |
| `FUSION_WEIGHT_INSAR` | `0.4` | Weight of InSAR signal in fused displacement |
| `ISOLATION_FOREST_CONTAMINATION` | `0.1` | Expected anomaly rate for Isolation Forest |
| `RANDOM_SEED` | `42` | Global random seed for reproducibility |
| `INSAR_IMAGE_MASK_THRESHOLD_QUANTILE` | `0.97` | Quantile for thresholding deformation in InSAR masks |

---

## 11. CLI Reference

**Entry point:** `python main.py`

```
usage: main.py [-h] [--skip-generate] [--skip-kaggle] [--run-agents]
               [--llm-provider {openai,groq,grok}] [--api-key API_KEY]
               [--llm-model LLM_MODEL]
```

| Flag | Default | Description |
|---|---|---|
| `--skip-generate` | off | Skip synthetic data generation; reuse existing files |
| `--skip-kaggle` | off | Skip Kaggle bridge model training |
| `--run-agents` | off | Run the CrewAI agent crew after the pipeline |
| `--llm-provider` | `openai` | LLM backend: `openai` or `gemini` |
| `--api-key` | from env | Override API key for the selected LLM provider |
| `--llm-model` | provider default | Override model name (e.g., `gpt-4o`, `gemini-2.5-pro`) |

### Example Commands

```bash
# Default run (generate synthetic data + train Kaggle model)
python main.py

# Fast re-run using existing data artifacts
python main.py --skip-generate

# Full pipeline + AI agent report using Gemini
python main.py --run-agents --llm-provider gemini --api-key YOUR_KEY

# Full pipeline + AI report with a specific model
python main.py --run-agents --llm-provider gemini --llm-model gemini/gemini-2.0-flash
```

### Console Output

A successful run prints:

```
Structural Health Monitoring pipeline completed
Rows: GNSS=N, InSAR=N, Sensors=N, Fused=N
Flags: GNSS threshold hits=N, InSAR threshold hits=N, Detected anomalies=N, InSAR frames=N
Kaggle model: rows=N, bridges=6, predicted_anomalies=N, threshold=0.XXXX
Kaggle metrics: precision=0.XXXX, recall=0.XXXX, f1=0.XXXX, pr_auc=0.XXXX
```

---

## 12. Environment Variables

Configure via the `.env` file in the project root (copy from `.env.example`).

| Variable | Required | Description |
|---|---|---|
| `CREWAI_PROVIDER` | When using agents | LLM provider identifier (`gemini`, `openai`) |
| `GEMINI_MODEL` | When using Gemini | Model string (e.g., `gemini/gemini-2.5-pro`) |
| `GEMINI_API_KEY` | When using Gemini | Google Gemini API key |
| `GOOGLE_API_KEY` | Alternative | Fallback if `GEMINI_API_KEY` is not set |
| `OPENAI_API_KEY` | When using OpenAI | OpenAI API key |
| `CREWAI_DISABLE_TELEMETRY` | Recommended | Set `true` to disable CrewAI usage tracking |

**Example `.env`:**
```env
CREWAI_PROVIDER=gemini
GEMINI_MODEL=gemini/gemini-2.5-pro
GEMINI_API_KEY=your_actual_api_key_here
CREWAI_DISABLE_TELEMETRY=true
```

> **Security:** Never commit your `.env` file. It is already listed in `.gitignore`.

---

## 13. Key ML Models

### Sensor Anomaly Model (`models/sensor_anomaly_model.pkl`)

| Property | Value |
|---|---|
| **Algorithm** | `IsolationForest` (scikit-learn) |
| **Preprocessing** | `StandardScaler` |
| **Contamination** | 0.1 |
| **Purpose** | Unsupervised anomaly detection on fused GNSS+InSAR+Sensor data |
| **Inputs** | All numeric columns from the fused training frame |
| **Output** | `anomaly_score` [0,1] and binary `anomaly` label |

### Bridge Anomaly Ensemble (`models/bridge_anomaly_model.pkl`)

| Property | Value |
|---|---|
| **Algorithm** | `StackingClassifier` |
| **Base estimators** | HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier |
| **Meta-learner** | LogisticRegression |
| **Preprocessing** | `ColumnTransformer` (median imputation for numeric, mode+OHE for categorical) |
| **Class balancing** | `compute_sample_weight("balanced")` on training |
| **Threshold** | Calibrated via F1-optimal precision-recall curve on validation set |
| **Purpose** | Supervised anomaly scoring on bridge sensor telemetry |
| **Features** | ~100+ engineered features from 16 core sensor signals |
| **Key metrics** | Precision, Recall, F1, PR-AUC, ROC-AUC |

### Model Validation Scorecard

The API computes a 5-check validation score (0–100) per bridge:

| Check | Weight | Source |
|---|---|---|
| **Model Quality Gate** | 30% | Precision + Recall + PR-AUC average |
| **Confidence Stability** | 25% | Peak and top-window mean anomaly probability |
| **Cross-Modal Agreement** | 20% | Number of supporting signal modalities agreeing |
| **Explainability Support** | 15% | Feature diversity, effective driver count, concentration index |
| **Hotspot Consistency** | 10% | Dominant anomaly zone share |

**Status mapping:**

| Score | Status |
|---|---|
| ≥ 80 | `verified` |
| 60–79 | `review` |
| < 60 | `weak` |

---

*Documentation generated from source code — April 2026.*
