# Structural Health Monitoring - Bridge Digital Twin

This project now supports a full **bridge-centric anomaly workflow** with:
- Kaggle dataset training
- Multi-sensor anomaly inference
- Map-based bridge selection
- 3D bridge visualization with anomaly markers
- GNSS, InSAR, sensor stream panels
- InSAR image mask predictions in frontend

## Dataset

Kaggle source:
- `mithil27360/digital-twin-bridge-structural-health-monitoring`

Downloaded file expected at:
- `data/external/bridge_digital_twin_dataset.csv`

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Set Kaggle token (PowerShell):

```powershell
$env:KAGGLE_API_TOKEN="<your_token>"
```

Download dataset (PowerShell):

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\Scripts\kaggle.exe" datasets download -d mithil27360/digital-twin-bridge-structural-health-monitoring -p data/external --unzip
```

## Train + Build Artifacts

Run full pipeline (synthetic + Kaggle):

```bash
python main.py
```

Skip synthetic and run only Kaggle model pipeline:

```bash
python main.py --skip-generate
```

Skip Kaggle and run only synthetic pipeline:

```bash
python main.py --skip-kaggle
```

## Frontend

Run:

```bash
streamlit run src/visualization/dashboard.py
```

## User Flow

1. User lands on dashboard.
2. User clicks a bridge marker on the map (or picks from selector fallback).
3. App auto-loads that bridge's GNSS/InSAR/sensor streams.
4. App runs the trained anomaly model for that bridge.
5. App renders anomalies on a 3D bridge digital twin.
6. User explores:
- GNSS lines
- InSAR LOS timeseries
- Sensor streams
- InSAR image + predicted mask + overlay
- Live anomaly table

## Generated Outputs

Global model outputs:
- `models/bridge_anomaly_model.pkl`
- `models/bridge_anomaly_metrics.json`
- `data/bridges/bridge_registry.csv`
- `data/bridges/bridge_predictions.csv`

Per bridge (for each bridge folder under `data/bridges/`):
- `source_dataset.csv`
- `gnss_raw.csv`
- `insar_timeseries.csv`
- `sensor_data.csv`
- `predictions.csv`
- `insar_mask_metadata.csv`
- `insar_images/*.png`
- `insar_masks/*.png`
- `insar_overlays/*.png`
