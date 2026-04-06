# Structural Health Monitoring - Complete Run Guide

## Prerequisites

1. **Python 3.10+** (3.11 recommended)
2. **Kaggle account** - for dataset download
3. **Ollama** (optional, for local AI agents) or **OpenAI API key**

## Step-by-Step Setup & Run

### 1. Navigate to project
```powershell
cd "c:/Users/Vijayakrishnaji/Desktop/capstone mainproject/Structural-Health-Monitoring"
```

### 2. Install dependencies (use existing virtualenv if available)
```powershell
# Activate existing env or create new
myenv\\Scripts\\activate

pip install -r requirements.txt
```

### 3. Download Kaggle dataset (one-time)
```powershell
# Set Kaggle token (get from kaggle.com/account)
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_api_key"

kaggle datasets download -d mithil27360/digital-twin-bridge-structural-health-monitoring -p data/external --unzip
```

**File created:** `data/external/bridge_digital_twin_dataset.csv`

### 4. Run full pipeline
```powershell
python main.py
```

**What happens:**
- ✅ Generates synthetic GNSS/InSAR/sensor data (with injected damage)
- ✅ Runs GNSS → InSAR → Sensor preprocessing & analysis
- ✅ Data fusion + Isolation Forest anomaly detection
- ✅ Kaggle bridge ML pipeline (6 digital twin bridges)
- ✅ Creates all models, CSVs, InSAR images/masks/overlays
- **Console output:** Row counts, anomaly stats, model metrics

**Time:** ~2-3 minutes first run

### 5. (Optional) Run with CrewAI Agents for analysis/reporting
```powershell
# Local Ollama (free, recommended)
python main.py --run-agents

# OpenAI (faster/better reports)
python main.py --run-agents --llm-provider openai --api-key $env:OPENAI_API_KEY
```

**What happens:**
- ✅ Runs full pipeline above
- ✅ 3 AI agents analyze results:
  1. **Data Analyst** → Summarizes thresholds/multi-modal agreement
  2. **Anomaly Triage** → Risk-ranks 6 bridges (HIGH/MEDIUM/LOW)
  3. **Report Writer** → Creates Markdown report
- **Output:** `reports/shm_report_YYYYMMDD_HHMMSS.md`

### 6. Launch the production web app
Backend:
```bash
.venv/bin/uvicorn src.api.main:app --reload
```

Frontend development server:
```bash
PATH="$(pwd)/.local/node/bin:$PATH" npm --prefix frontend run dev
```

Production build:
```bash
PATH="$(pwd)/.local/node/bin:$PATH" npm --prefix frontend run build
.venv/bin/uvicorn src.api.main:app
```

**Open:** http://127.0.0.1:8000

**Features:**
- Fleet overview with risk map and priority queue
- Per-bridge telemetry workspace for GNSS, InSAR, and sensor data
- Bridge digital twin hotspot visualization
- InSAR frame viewer with amplitude, interferogram, heatmap, coherence, and overlay images
- Explainability factors, anomaly ledger, and markdown report browser
- FastAPI endpoints for bridge refresh, full pipeline rerun, and report generation

## Quick Commands Summary

| Goal | Command |
|---|---|
| **Full pipeline** | `python main.py` |
| **Pipeline + AI agents** | `python main.py --run-agents` |
| **Fast re-run (skip data gen)** | `python main.py --skip-generate` |
| **Web API + built frontend** | `.venv/bin/uvicorn src.api.main:app` |
| **Frontend dev server** | `PATH="$(pwd)/.local/node/bin:$PATH" npm --prefix frontend run dev` |
| **Agents w/ OpenAI** | `python main.py --run-agents --llm-provider openai --api-key sk-...` |

## Expected Outputs

```
data/bridges/bridge_alpha/          ← 6 bridge folders (gnss, insar, predictions, XAI)
data/fused/fused_displacement.csv   ← Fused multi-modal anomalies
models/bridge_anomaly_model.pkl     ← Trained ensemble model
models/sensor_anomaly_model.pkl     ← Isolation Forest model
reports/shm_report_*.md            ← AI-generated engineering report (agents only)
```

## Troubleshooting

| Issue | Solution |
|---|---|
| **Kaggle auth** | Set `$env:KAGGLE_USERNAME` + `$env:KAGGLE_KEY` |
| **Missing dataset** | Run kaggle download command |
| **Ollama agents slow** | Use `--llm-provider openai` |
| **No reports/ dir** | Agents auto-create it |
| **Frontend build fails** | `PATH="$(pwd)/.local/node/bin:$PATH" npm --prefix frontend install` |
| **API dependencies missing** | `pip install -r requirements.txt` |
| **CUDA out of memory** | Use CPU-only: `pip uninstall torch` |

## Verify Success

✅ **Pipeline:** Check `data/bridges/` has 6 folders, models/*.pkl exist  
✅ **Agents:** Check `reports/shm_report_*.md` with risk-ranked bridges  
✅ **Web app:** http://127.0.0.1:8000 shows the control room interface  

**Project fully operational! 🚀**
