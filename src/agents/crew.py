from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

ProgressCallback = Callable[[dict[str, Any]], None]


class ExecutiveNarrative(BaseModel):
    headline: str
    fleet_status: str
    summary_points: list[str] = Field(default_factory=list)
    watchlist: list[str] = Field(default_factory=list)
    model_commentary: str
    operational_note: str


class PriorityBridge(BaseModel):
    bridge_id: str
    bridge_name: str
    risk_level: str
    urgency_score: int
    recommendation: str
    rationale: str
    dominant_hotspot: str
    primary_driver: str
    modalities_agreeing: list[str] = Field(default_factory=list)
    anomaly_count: int
    max_probability: float


class TriageSummary(BaseModel):
    escalation_headline: str
    bridge_priorities: list[PriorityBridge] = Field(default_factory=list)
    action_counts: dict[str, int] = Field(default_factory=dict)
    consensus_notes: list[str] = Field(default_factory=list)
    maintenance_queue: list[str] = Field(default_factory=list)


class DashboardVisualPlan(BaseModel):
    hero_title: str
    hero_message: str
    panel_callouts: list[str] = Field(default_factory=list)
    chart_annotations: list[str] = Field(default_factory=list)
    visual_watchouts: list[str] = Field(default_factory=list)
    operator_prompts: list[str] = Field(default_factory=list)


class ReportPacket(BaseModel):
    report_title: str
    executive_brief: str
    admin_recommendations: list[str] = Field(default_factory=list)
    markdown_report: str


@dataclass
class CrewKickoffResult:
    artifact_name: str
    markdown_content: str
    artifact: dict[str, Any]


def _notify(progress_callback: ProgressCallback | None, stage: str, detail: str, **meta: Any) -> None:
    if progress_callback:
        progress_callback({"stage": stage, "detail": detail, "meta": meta})


def _json_safe_output(task: Any) -> dict[str, Any]:
    output = getattr(task, "output", None)
    if output is None:
        return {}

    for attr in ("json_dict", "pydantic"):
        value = getattr(output, attr, None)
        if value:
            if hasattr(value, "model_dump"):
                return value.model_dump()
            if isinstance(value, dict):
                return value

    raw = getattr(output, "raw", None)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
    if isinstance(raw, dict):
        return raw
    return {}


def _get_llm(provider: str, api_key: str | None = None, model: str | None = None):
    from crewai import LLM

    provider_key = provider.lower().strip()
    if provider_key in {"gemini", "google", "google-genai"}:
        model_name = model or os.environ.get("GEMINI_MODEL") or "gemini/gemini-2.5-pro"
        api_key_to_use = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key_to_use:
            raise ValueError("Gemini API key is required. Provide it from the Reports panel or set GEMINI_API_KEY.")
        return LLM(model=model_name, api_key=api_key_to_use)

    if provider_key == "openai":
        model_name = model or "gpt-4o-mini"
        api_key_to_use = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key_to_use:
            raise ValueError("OpenAI API key is required. Provide it or set OPENAI_API_KEY.")
        return LLM(model=model_name, api_key=api_key_to_use)

    raise ValueError(f"Unknown LLM provider: {provider}")


def kickoff_shm_crew_dashboard(
    pipeline_summary: dict[str, Any],
    *,
    provider: str = "gemini",
    api_key: str | None = None,
    model: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CrewKickoffResult:
    project_root = Path(__file__).resolve().parents[2]
    crew_home = project_root / ".cache" / "crewai_home"
    crew_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(crew_home)
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ["CREWAI_STORAGE_DIR"] = "Capstone_Vijay"

    try:
        from crewai import Agent, Crew, Process, Task
        from src.agents.tools import (
            ReadBridgePredictionsTool,
            ReadBridgeTelemetryTool,
            ReadBridgeXAITool,
            ReadFleetOverviewTool,
            ReadModelMetricsTool,
            WriteReportTool,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "CrewAI is not installed in this environment. Install `crewai[google-genai]` to enable the Reports workspace."
        ) from exc

    llm = _get_llm(provider, api_key, model)
    _notify(progress_callback, "Crew Boot", "CrewAI runtime initialized and Gemini session prepared.", provider=provider, model=model or os.environ.get("GEMINI_MODEL") or "gemini/gemini-2.5-pro")

    all_bridges = pipeline_summary.get("overview", {}).get("bridges", [])[:6]
    focus_bridge_ids = [bridge.get("bridge_id") for bridge in all_bridges if bridge.get("bridge_id")]
    focus_bridge_names = [bridge.get("bridge_name") for bridge in all_bridges if bridge.get("bridge_name")]

    fleet_tool = ReadFleetOverviewTool(progress_callback=progress_callback)
    metrics_tool = ReadModelMetricsTool(progress_callback=progress_callback)
    predictions_tool = ReadBridgePredictionsTool(progress_callback=progress_callback)
    telemetry_tool = ReadBridgeTelemetryTool(progress_callback=progress_callback)
    xai_tool = ReadBridgeXAITool(progress_callback=progress_callback)
    write_tool = WriteReportTool(progress_callback=progress_callback)

    analyst = Agent(
        role="Fleet Intelligence Analyst",
        goal="Convert raw SHM telemetry and model metrics into admin-ready operational intelligence.",
        backstory=(
            "You are the lead infrastructure analytics specialist for a transportation operations command center. "
            "You excel at turning raw bridge telemetry, model metrics, and fleet outputs into concise, factual, executive-safe language."
        ),
        tools=[fleet_tool, metrics_tool, telemetry_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    triage = Agent(
        role="Structural Triage Engineer",
        goal="Rank bridges by operational risk, explain why they are risky, and prescribe admin actions.",
        backstory=(
            "You are a senior bridge triage engineer trusted by district administrators. "
            "You distinguish between noisy alerts and true intervention candidates, and you always explain the operational reason behind each action."
        ),
        tools=[predictions_tool, xai_tool, telemetry_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    visual_strategist = Agent(
        role="Analytics Dashboard Strategist",
        goal="Translate engineering findings into an executive analytics dashboard narrative with clear visual callouts.",
        backstory=(
            "You design admin dashboards for infrastructure command centers. "
            "You do not write code, but you specify what decision-makers should immediately see, compare, and act on."
        ),
        tools=[fleet_tool, predictions_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    reporter = Agent(
        role="Executive Reporting Lead",
        goal="Produce a formal engineering briefing that matches the admin dashboard and can be archived for audit and stakeholder review.",
        backstory=(
            "You write high-stakes operational briefs for transportation leadership. "
            "You blend technical facts with clear escalation guidance and preserve traceability for later review."
        ),
        tools=[write_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    analyst_task = Task(
        description=(
            "You are preparing the operational intelligence narrative for the bridge monitoring admin workspace. "
            "Use the fleet overview tool, model metrics tool, and the bridge telemetry tool for all six monitored bridges "
            f"{focus_bridge_ids}. Return strict JSON only matching the schema. "
            f"Pipeline summary context: {json.dumps(pipeline_summary)}"
        ),
        expected_output="Strict JSON summarizing fleet health, watchlist bridges, operational notes, and model commentary.",
        agent=analyst,
        output_json=ExecutiveNarrative,
    )

    triage_task = Task(
        description=(
            "Rank the active bridges for intervention. Use the bridge predictions tool and the XAI tool for focus bridges "
            f"{focus_bridge_ids}. Also use the telemetry tool when you need to verify cross-modality agreement. "
            "Return strict JSON only with bridge priorities, action counts, consensus notes, and maintenance queue."
        ),
        expected_output="Strict JSON containing ranked bridge priorities and intervention recommendations.",
        agent=triage,
        output_json=TriageSummary,
        context=[analyst_task],
    )

    visual_task = Task(
        description=(
            "Design the analytics story for the admin dashboard. Use the fleet overview and bridge predictions tools. "
            "You must convert the analyst and triage findings into a visual plan that highlights what leadership should see first, "
            "which callouts belong beside charts, and what operator prompts belong in the workspace. Return strict JSON only."
        ),
        expected_output="Strict JSON with hero messaging, chart annotations, dashboard callouts, and operator prompts.",
        agent=visual_strategist,
        output_json=DashboardVisualPlan,
        context=[analyst_task, triage_task],
    )

    report_task = Task(
        description=(
            "Write the archival engineering brief for this run. Use the prior task outputs as your source of truth. "
            "Return strict JSON only containing a report title, executive brief, admin recommendations, and the final markdown report. "
            "The markdown should include sections for executive summary, bridge priorities, visual highlights, and maintenance actions."
        ),
        expected_output="Strict JSON with a polished markdown report packet.",
        agent=reporter,
        output_json=ReportPacket,
        context=[analyst_task, triage_task, visual_task],
    )

    crew = Crew(
        agents=[analyst, triage, visual_strategist, reporter],
        tasks=[analyst_task, triage_task, visual_task, report_task],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True,
    )

    _notify(progress_callback, "Crew Dispatch", "Crew manager is assigning analyst, triage, dashboard, and reporting tasks.")
    crew.kickoff()
    _notify(progress_callback, "Crew Synthesis", "CrewAI tasks completed. Consolidating dashboard artifacts.")

    analyst_output = _json_safe_output(analyst_task)
    triage_output = _json_safe_output(triage_task)
    visual_output = _json_safe_output(visual_task)
    report_output = _json_safe_output(report_task)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    artifact_name = f"crew_dashboard_{timestamp}"

    markdown_content = report_output.get("markdown_report") or "\n".join(
        [
            f"# {report_output.get('report_title', 'CrewAI SHM Dashboard Brief')}",
            "",
            report_output.get("executive_brief", ""),
        ]
    )

    writer_payload = json.loads(write_tool._run(markdown_content=markdown_content, file_stem=artifact_name))
    _notify(progress_callback, "Crew Archive", "CrewAI markdown report archived for the Reports workspace.", path=writer_payload.get("path"))

    artifact = {
        "metadata": {
            "name": artifact_name,
            "generated_at": datetime.utcnow().isoformat(),
            "provider": provider,
            "model": model or os.environ.get("GEMINI_MODEL") or "gemini/gemini-2.5-pro",
            "focus_bridges": focus_bridge_names,
            "focus_bridge_ids": focus_bridge_ids,
            "overview_bridges": pipeline_summary.get("overview", {}).get("bridges", []),
            "markdown_path": writer_payload.get("path"),
        },
        "analyst": analyst_output,
        "triage": triage_output,
        "visual_plan": visual_output,
        "report": report_output,
    }
    return CrewKickoffResult(artifact_name=artifact_name, markdown_content=markdown_content, artifact=artifact)
