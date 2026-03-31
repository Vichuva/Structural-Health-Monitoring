# CrewAI Integration Suggestions
## Structural Health Monitoring — Bridge Digital Twin

---

## Why CrewAI Fits This Project

The current pipeline is a **linear script** — it runs steps sequentially and produces outputs, but it cannot reason, explain, or decide. CrewAI enables a **multi-agent layer** on top of the existing pipeline where specialized AI agents collaborate to interpret sensor data, prioritize alerts, and generate human-readable engineering reports — without changing any of the existing pipeline code.

---

## Proposed Agent Crew

### Overview

```
                    ┌──────────────────┐
                    │   Crew Manager   │
                    │  (Process: hier- │
                    │    archical)     │
                    └────────┬─────────┘
                             │ delegates to
          ┌──────────────────┼───────────────────┐
          ▼                  ▼                   ▼
  ┌───────────────┐  ┌──────────────┐  ┌─────────────────┐
  │  Data Analyst │  │   Anomaly    │  │    Reporting    │
  │    Agent      │  │  Triage Agent│  │     Agent       │
  └───────────────┘  └──────────────┘  └─────────────────┘
          │                  │                   │
          ▼                  ▼                   ▼
  Reads pipeline      Ranks alerts by      Writes natural
  outputs, checks     severity, assigns    language report
  thresholds,         bridge risk level,   + maintenance
  summarizes data      recommends action    recommendations
```

---

## Suggested Agents and Their Roles

### Agent 1 — Data Analyst Agent

**Role:** Reads and interprets all pipeline output CSV files after `python main.py` completes.

**Where it fits:** After the full pipeline runs (end of `run_pipeline()` in `main.py`), this agent is triggered to summarize what happened.

**What it does:**
- Reads `gnss_analysis.csv` → counts and explains threshold exceedances in plain English
- Reads `insar_analysis.csv` → summarizes LOS displacement trends
- Reads `sensor_features.csv` → detects which sensor channel spiked most
- Reads `fused_displacement.csv` → gives a holistic displacement narrative
- Reads `bridge_anomaly_metrics.json` → interprets precision/recall/F1 in plain language

**Useful because:** Currently the pipeline only prints raw numbers. This agent turns them into: *"Vertical GNSS displacement exceeded the 5 mm threshold on 12 out of 60 days. The anomaly pattern began on Day 40 — consistent with progressive settlement."*

---

### Agent 2 — Anomaly Triage Agent

**Role:** Acts as a structural engineer's assistant. Takes the anomaly output and decides which bridges or time windows need urgent attention.

**Where it fits:** After `run_anomaly_detection()` and `run_kaggle_bridge_pipeline()` complete. Reads from `bridge_predictions.csv` and `fused_displacement.csv`.

**What it does:**
- Ranks the 6 bridges by anomaly risk (High / Medium / Low)
- For each high-risk bridge, identifies which anomaly driver (from `xai_top_factors.csv`) is most critical
- Checks whether multiple modalities agree (GNSS + InSAR + sensor all flagging) vs. a single-modality flag (lower confidence)
- Recommends one of three actions: `Immediate Inspection`, `Scheduled Monitoring`, `No Action`

**Useful because:** Currently the pipeline outputs rows of `anomaly=1` with no prioritization. This agent makes the output actionable and decision-ready for a maintenance team.

**Example output:**
```
🔴 Bridge: Pacific Crown (San Francisco) — HIGH RISK
   - 3 modalities agree: GNSS vertical = 9.2 mm, InSAR LOS = 11 mm, Vibration spike +84%
   - Top driver: Probability_of_Failure_PoF (XAI impact: +0.42)
   - Recommendation: IMMEDIATE INSPECTION

🟡 Bridge: Sound Span (Seattle) — MEDIUM RISK
   - Only sensor data flagged. GNSS and InSAR within normal range.
   - Recommendation: Increase monitoring frequency to weekly.
```

---

### Agent 3 — Report Generation Agent

**Role:** Synthesizes the outputs of Agent 1 and Agent 2 into a professional engineering report in Markdown or PDF format.

**Where it fits:** As the final step — after both other agents complete their analysis.

**What it does:**
- Writes an executive summary (non-technical, suitable for facility managers)
- Writes a technical section (for engineers): full metrics, threshold counts, model performance
- Embeds the InSAR overlay images and 3D anomaly location as references
- Generates a timestamped report file: `reports/shm_report_YYYYMMDD.md`

**Useful because:** Currently there is no persistent human-readable output. A report that non-technical stakeholders can read is a key gap in the pipeline.

---

### Agent 4 — Maintenance Scheduler Agent *(Optional, Advanced)*

**Role:** Uses the triage output and historical reports to recommend an optimal inspection schedule.

**Where it fits:** As an optional agent triggered only when `--schedule` CLI flag is passed.

**What it does:**
- Reads historical reports (if multiple pipeline runs exist)
- Tracks trend of anomaly counts per bridge over time
- Suggests inspection dates based on rate of deterioration
- Issues early warnings if a bridge's health is declining faster than expected

**Useful because:** Transforms the system from a one-shot detector into a longitudinal health tracker that proactively schedules maintenance before failures occur.

---

## Where Exactly to Integrate in the Codebase

| Integration Point | File | What to Add |
|---|---|---|
| After `run_pipeline()` returns `summary` | `main.py` line 44 | Trigger `DataAnalystAgent` with the summary dict |
| After `run_anomaly_detection()` | `main.py` line 31 | Pass `anomalies` DataFrame to `TriageAgent` |
| After `run_kaggle_bridge_pipeline()` | `main.py` line 50 | Pass Kaggle `summary` to `TriageAgent` |
| New file | `src/agents/crew.py` | Define CrewAI crew, agents, tasks |
| New CLI flag | `main.py` argparse | Add `--run-agents` flag to trigger crew |
| New output folder | `reports/` | Store all agent-generated reports |

---

## Suggested New Files to Create

```
src/
└── agents/
    ├── __init__.py
    ├── crew.py              ← Defines the Crew, all Agents, all Tasks
    ├── tools.py             ← Custom CrewAI tools that read pipeline CSVs
    └── prompts.py           ← Prompt templates for each agent role
reports/                     ← Auto-generated markdown/PDF reports
```

---

## Suggested CrewAI Tools (Custom)

CrewAI agents use **Tools** to interact with the outside world. These would be custom tools wrapping the existing pipeline outputs:

| Tool Name | What It Does |
|---|---|
| `ReadGNSSAnalysisTool` | Reads `gnss_analysis.csv`, returns summary stats |
| `ReadBridgePredictionsTool` | Reads `bridge_predictions.csv`, returns top anomaly bridges |
| `ReadXAIFactorsTool` | Reads `xai_top_factors.csv` for a given bridge_id |
| `ReadModelMetricsTool` | Reads `bridge_anomaly_metrics.json` |
| `ReadInSARMaskMetaTool` | Reads `insar_mask_metadata.csv`, returns mask ratio trend |
| `WriteReportTool` | Writes the final report to `reports/shm_report_<date>.md` |

---

## Process Type Recommendation

Use **Hierarchical Process** in CrewAI (not sequential):
- A **Manager LLM** decides which agent to call and in what order
- This allows the Triage Agent to ask the Data Analyst Agent for more detail before making its recommendation
- More robust than sequential — if one piece of data is missing, the manager can skip that step gracefully

---

## Suggested LLM

- **Local (free):** `ollama` with `llama3` or `mistral` — no API cost, runs on-device
- **Cloud:** `gpt-4o-mini` (OpenAI) — cheap and highly capable for structured text analysis
- **Best for this project:** `gpt-4o-mini` for the report agent (needs good writing); `ollama/mistral` for the data analyst (structured parsing is simpler)

---

## Summary of Value Added by CrewAI

| Before (Current) | After (With CrewAI) |
|---|---|
| Raw numbers printed to terminal | Natural language narratives |
| No prioritization of alerts | Risk-ranked bridge list |
| No recommendations | Action items per bridge |
| No persistent output | Timestamped professional report |
| Engineers must interpret results | Results are pre-interpreted |
| One-shot run | Optional longitudinal tracking |

---

## Recommended Implementation Order

1. Create `src/agents/tools.py` — wrap CSV readers as CrewAI tools
2. Create `src/agents/crew.py` — define agents and tasks
3. Add `--run-agents` flag to `main.py` argparse
4. Call `crew.kickoff()` at the end of `main()` when flag is set
5. Add `reports/` folder and `.gitignore` entry for generated reports
6. (Optional) Add `--schedule` flag + Maintenance Scheduler Agent

---

*This document outlines suggestions only. No code has been implemented.*
