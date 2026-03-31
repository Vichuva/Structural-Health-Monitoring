import json
from pathlib import Path
from crewai import Agent, Task, Crew, Process

from src.agents.prompts import (
    DATA_ANALYST_ROLE,
    DATA_ANALYST_GOAL,
    DATA_ANALYST_BACKSTORY,
    TRIAGE_ROLE,
    TRIAGE_GOAL,
    TRIAGE_BACKSTORY,
    REPORT_WRITER_ROLE,
    REPORT_WRITER_GOAL,
    REPORT_WRITER_BACKSTORY,
)
from src.agents.tools import (
    ReadGNSSAnalysisTool,
    ReadBridgePredictionsTool,
    ReadXAIFactorsTool,
    ReadModelMetricsTool,
    WriteReportTool,
)

from crewai import LLM

def _get_llm(provider: str, api_key: str = None, model: str | None = None):
    import os
    if provider == "openai":
        model_name = model or "gpt-4o-mini"
        api_key_to_use = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key_to_use:
            raise ValueError("OpenAI API key is required. Pass it as --api-key or set OPENAI_API_KEY environment variable.")
        return LLM(model=model_name, api_key=api_key_to_use)
    elif provider == "groq":
        model="groq/llama-3.3-70b-versatile"
        api_key_to_use = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key_to_use:
            raise ValueError("Groq API key is required. Pass it as --api-key or set GROQ_API_KEY environment variable.")
        return LLM(model, api_key=api_key_to_use)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

def kickoff_shm_crew(pipeline_summary: dict, provider: str = "openai", api_key: str = None, model: str | None = None):
    """    Kicks off the multi-agent CrewAI process that reads the pipeline's output
    and synthesizes an engineering report.
    """
    llm = _get_llm(provider, api_key, model)
    
    # 1. Define Tools
    tool_gnss = ReadGNSSAnalysisTool()
    tool_preds = ReadBridgePredictionsTool()
    tool_xai = ReadXAIFactorsTool()
    tool_metrics = ReadModelMetricsTool()
    tool_write = WriteReportTool()
    
    # 2. Define Agents
    analyst = Agent(
        role=DATA_ANALYST_ROLE,
        goal=DATA_ANALYST_GOAL,
        backstory=DATA_ANALYST_BACKSTORY,
        tools=[tool_gnss, tool_metrics],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    triage = Agent(
        role=TRIAGE_ROLE,
        goal=TRIAGE_GOAL,
        backstory=TRIAGE_BACKSTORY,
        tools=[tool_preds, tool_xai],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    reporter = Agent(
        role=REPORT_WRITER_ROLE,
        goal=REPORT_WRITER_GOAL,
        backstory=REPORT_WRITER_BACKSTORY,
        tools=[tool_write],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # 3. Define Tasks
    task_analyze_data = Task(
        description=f"Review the GNSS dataset and the model metrics for the latest pipeline run. Summarize the overall health of the entire monitoring fleet. Here is the high-level summary of the pipeline run that just finished: {json.dumps(pipeline_summary)}. Extract key factual numbers.",
        expected_output="A concise factual summary of the fleet's GNSS displacement metrics and the overall ML model performance.",
        agent=analyst
    )
    
    task_triage = Task(
        description="Check all the bridges for anomaly predictions. Identify which specific bridges are flagging as anomalous. Then, map those alerts to the XAI factors (root cause drivers) for each anomalous bridge. Rank the severity (High/Med/Low) based on how many modalities flag an error.",
        expected_output="A risk-ranked list of anomalous bridges, including the top factors driving the alert and recommended maintenance actions.",
        agent=triage
    )
    
    task_report = Task(
        description="Using the overall data analysis from the Analyst AND the risk-ranked lists from the Triage Engineer, write an executive Markdown report. Include a non-technical summary at the top. Finally, strictly USE your WriteReportTool to save the Markdown content to a file. Pass the exact Markdown to the tool.",
        expected_output="Confirmation that the Markdown engineering report has been written successfully to disk via the WriteReportTool.",
        agent=reporter
    )
    
    # 4. Form Crew and Execute
    crew = Crew(
        agents=[analyst, triage, reporter],
        tasks=[task_analyze_data, task_triage, task_report],
        process=Process.sequential,
        verbose=True
    )
    
    print("\n[CrewAI] Initiating multi-agent collaboration...")
    result = crew.kickoff()
    return result
