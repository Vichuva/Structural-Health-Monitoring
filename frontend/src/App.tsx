import { Canvas } from "@react-three/fiber";
import { Html, OrbitControls, RoundedBox } from "@react-three/drei";
import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { api } from "./lib/api";
import type {
  AnomalyRow,
  BridgeCard,
  BridgeDetail,
  Hotspot,
  InSarFrame,
  OperationStatus,
  OverviewResponse,
  ReportDetail,
  RuntimeStage,
  TimePoint,
  XaiFactor
} from "./types";

type TelemetryMode = "gnss" | "insar" | "sensors";
type WorkspaceSection = "executive" | "bridge" | "telemetry" | "insar" | "reports";

const sections: Array<{ id: WorkspaceSection; label: string; hint: string }> = [
  { id: "executive", label: "Executive", hint: "Fleet posture and risk overview" },
  { id: "bridge", label: "Bridge Ops", hint: "3D twin and runtime operations" },
  { id: "telemetry", label: "Telemetry Lab", hint: "GNSS, InSAR, and sensor analytics" },
  { id: "insar", label: "InSAR Explorer", hint: "Frame-by-frame deformation imagery" },
  { id: "reports", label: "Reports", hint: "Engineering reports and writeups" }
];

type TowerScene = {
  x: number;
  height: number;
  width: number;
};

type PierScene = {
  x: number;
  height: number;
  width: number;
};

type BridgeSceneConfig = {
  deckLength: number;
  deckWidth: number;
  deckThickness: number;
  deckY: number;
  groupPosition: [number, number, number];
  groupRotation: [number, number, number];
  cameraPosition: [number, number, number];
  cameraFov: number;
  cameraTarget: [number, number, number];
  minDistance: number;
  maxDistance: number;
  piers: PierScene[];
  towers: TowerScene[];
  cableMode: "fan" | "hybrid" | "suspension";
  zoneAnchors: Record<string, Array<[number, number, number]>>;
};

type PositionedAnomaly = {
  id: string;
  position: [number, number, number];
  severity: number;
  row: AnomalyRow;
  title: string;
  detail: string;
};

type PositionedHotspot = {
  id: string;
  position: [number, number, number];
  hotspot: Hotspot;
};

function App() {
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [detail, setDetail] = useState<BridgeDetail | null>(null);
  const [selectedBridgeId, setSelectedBridgeId] = useState("");
  const [selectedReport, setSelectedReport] = useState("");
  const [report, setReport] = useState<ReportDetail | null>(null);
  const [telemetryMode, setTelemetryMode] = useState<TelemetryMode>("gnss");
  const [section, setSection] = useState<WorkspaceSection>("executive");
  const [insarIndex, setInsarIndex] = useState(0);
  const [loadingOverview, setLoadingOverview] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [actionState, setActionState] = useState("");
  const [activeOperation, setActiveOperation] = useState<OperationStatus | null>(null);
  const [operationMinimized, setOperationMinimized] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    void loadOverview();
  }, []);

  useEffect(() => {
    if (selectedBridgeId) {
      void loadBridge(selectedBridgeId);
    }
  }, [selectedBridgeId]);

  useEffect(() => {
    if (selectedReport) {
      void loadReport(selectedReport);
    }
  }, [selectedReport]);

  useEffect(() => {
    if (!activeOperation || activeOperation.status === "completed" || activeOperation.status === "failed") {
      return;
    }

    const interval = window.setInterval(async () => {
      try {
        const next = await api.getOperation(activeOperation.id);
        setActiveOperation(next);
        setActionState(next.message);

        if (next.status === "completed") {
          if (next.kind === "bridge_refresh" && selectedBridgeId) {
            await loadBridge(selectedBridgeId);
            setOverview(await api.getOverview());
          }
          if (next.kind === "full_pipeline") {
            await loadOverview();
            if (selectedBridgeId) {
              await loadBridge(selectedBridgeId);
            }
          }
          if (next.kind === "crew_report") {
            const refreshedOverview = await api.getOverview();
            setOverview(refreshedOverview);
            const generatedName = String((next.result as { report?: { name?: string } } | null)?.report?.name || "");
            if (generatedName) {
              setSelectedReport(generatedName);
              await loadReport(generatedName);
            }
          }
          setActionState("");
          setOperationMinimized(false);
        }

        if (next.status === "failed") {
          setError(next.error || next.message);
          setActionState("");
        }
      } catch (err) {
        setError(getErrorMessage(err));
        setActionState("");
      }
    }, 1200);

    return () => window.clearInterval(interval);
  }, [activeOperation, selectedBridgeId]);

  async function loadOverview() {
    try {
      setLoadingOverview(true);
      setError("");
      const payload = await api.getOverview();
      setOverview(payload);
      if (!selectedBridgeId && payload.bridges.length > 0) {
        setSelectedBridgeId(payload.bridges[0].bridge_id);
      }
      if (!selectedReport && payload.reports.length > 0) {
        setSelectedReport(payload.reports[0].name);
      }
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoadingOverview(false);
    }
  }

  async function loadBridge(bridgeId: string) {
    try {
      setLoadingDetail(true);
      setError("");
      const payload = await api.getBridgeDetail(bridgeId);
      setDetail(payload);
      setInsarIndex(0);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoadingDetail(false);
    }
  }

  async function loadReport(reportName: string) {
    try {
      setError("");
      setReport(await api.getReport(reportName));
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }

  async function handleBridgeRefresh() {
    if (!selectedBridgeId) {
      return;
    }
    try {
      setError("");
      const operation = await api.refreshBridge(selectedBridgeId);
      setActiveOperation(operation);
      setOperationMinimized(false);
      setActionState(operation.message);
      setSection("bridge");
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }

  async function handlePipelineRun() {
    try {
      setError("");
      const operation = await api.runPipeline();
      setActiveOperation(operation);
      setOperationMinimized(false);
      setActionState(operation.message);
      setSection("executive");
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }

  async function handleGenerateReport() {
    try {
      setActionState("Starting CrewAI analytics run...");
      const operation = await api.generateReport();
      setActiveOperation(operation);
      setOperationMinimized(false);
      setSection("reports");
    } catch (err) {
      setError(getErrorMessage(err));
    }
  }

  const selectedBridgeCard = useMemo(
    () => overview?.bridges.find((bridge) => bridge.bridge_id === selectedBridgeId) ?? null,
    [overview, selectedBridgeId]
  );

  return (
    <div className="workspace-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <span className="kicker">Command Center</span>
          <h1>Structural Health Operations</h1>
          <p>Production-grade bridge monitoring workspace for executive review, field triage, and forensic analytics.</p>
        </div>

        <nav className="section-nav">
          {sections.map((item) => (
            <button
              key={item.id}
              className={`section-nav-item ${section === item.id ? "active" : ""}`}
              onClick={() => setSection(item.id)}
            >
              <strong>{item.label}</strong>
              <span>{item.hint}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-group">
          <div className="sidebar-heading">
            <span>Bridge Portfolio</span>
            <strong>{overview?.bridges.length ?? 0}</strong>
          </div>
          <div className="bridge-stack">
            {loadingOverview && <PanelPlaceholder label="Loading fleet registry" />}
            {overview?.bridges.map((bridge) => (
              <button
                key={bridge.bridge_id}
                className={`bridge-list-item ${bridge.bridge_id === selectedBridgeId ? "active" : ""}`}
                onClick={() => {
                  setSelectedBridgeId(bridge.bridge_id);
                  setSection("bridge");
                }}
              >
                <div>
                  <strong>{bridge.bridge_name}</strong>
                  <span>
                    {bridge.city}, {bridge.region}
                  </span>
                </div>
                <div className="bridge-badges">
                  <small>{bridge.top_hotspot}</small>
                  <em>{formatPercent(bridge.max_probability)}</em>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="sidebar-group">
          <div className="sidebar-heading">
            <span>Reports</span>
            <button className="plain-button" onClick={() => void handleGenerateReport()}>
              Generate
            </button>
          </div>
          <div className="report-stack">
            {overview?.reports.map((entry) => (
              <button
                key={entry.name}
                className={`report-list-item ${entry.name === selectedReport ? "active" : ""}`}
                onClick={() => {
                  setSelectedReport(entry.name);
                  setSection("reports");
                }}
              >
                <strong>{entry.title}</strong>
                <span>{formatDate(entry.updated_at)}</span>
              </button>
            ))}
          </div>
        </div>
      </aside>

      <main className="workspace-main">
        <header className="topbar">
          <div>
            <span className="kicker">Administrative Analytics Workspace</span>
            <h2>{sectionTitle(section)}</h2>
            <p>{sectionCopy(section, selectedBridgeCard?.bridge_name ?? "the selected bridge")}</p>
          </div>
          <div className="topbar-actions">
            <button className="button-secondary" onClick={() => void handleBridgeRefresh()} disabled={!selectedBridgeId}>
              Refresh Bridge
            </button>
            <button className="button-primary" onClick={() => void handlePipelineRun()}>
              Run Full Pipeline
            </button>
          </div>
        </header>

        {actionState && <div className="notice info">{actionState}</div>}
        {error && <div className="notice error">{error}</div>}
        {activeOperation && (
          <OperationPanel
            operation={activeOperation}
            minimized={operationMinimized}
            onToggleMinimize={() => setOperationMinimized((value) => !value)}
          />
        )}

        {section === "executive" && overview && (
          <ExecutiveView
            overview={overview}
            selectedBridgeId={selectedBridgeId}
            onSelectBridge={setSelectedBridgeId}
          />
        )}

        {section === "bridge" && (
          <BridgeOperationsView
            detail={detail}
            loading={loadingDetail}
            selectedBridgeName={selectedBridgeCard?.bridge_name ?? "Bridge"}
          />
        )}

        {section === "telemetry" && (
          <TelemetryView
            detail={detail}
            loading={loadingDetail}
            telemetryMode={telemetryMode}
            onTelemetryModeChange={setTelemetryMode}
          />
        )}

        {section === "insar" && (
          <InsarView
            detail={detail}
            loading={loadingDetail}
            index={insarIndex}
            onIndexChange={setInsarIndex}
          />
        )}

        {section === "reports" && (
          <ReportsView
            report={report}
            reports={overview?.reports ?? []}
            selectedReport={selectedReport}
            onSelectReport={setSelectedReport}
            onGenerateReport={() => void handleGenerateReport()}
            activeOperation={activeOperation?.kind === "crew_report" ? activeOperation : null}
          />
        )}
      </main>
    </div>
  );
}

function ExecutiveView({
  overview,
  selectedBridgeId,
  onSelectBridge
}: {
  overview: OverviewResponse;
  selectedBridgeId: string;
  onSelectBridge: (bridgeId: string) => void;
}) {
  return (
    <div className="view-grid">
      <section className="metric-band">
        <InfoCard label="Monitored Bridges" value={String(overview.fleet_metrics.total_bridges)} detail="Total digital twins currently in the monitoring program." />
        <InfoCard label="Bridges With Alerts" value={String(overview.fleet_metrics.bridges_with_alerts)} detail="Assets currently crossing active anomaly thresholds." />
        <InfoCard label="Peak Fleet Probability" value={formatPercent(overview.fleet_metrics.peak_probability)} detail="Maximum anomaly probability across the entire monitored fleet." />
        <InfoCard label="Highest Risk Asset" value={overview.fleet_metrics.highest_risk_bridge} detail="Bridge ranked highest by current risk posture." />
      </section>

      <section className="content-grid executive-grid">
        <div className="panel hero-surface">
          <PanelHeader title="Fleet Risk Matrix" subtitle="Portfolio-level visibility for rapid admin decision-making." />
          <FleetMap bridges={overview.bridges} selectedBridgeId={selectedBridgeId} onSelect={onSelectBridge} />
        </div>

        <div className="stack-panel">
          <div className="panel">
            <PanelHeader title="Model Confidence" subtitle="Current anomaly model health metrics." />
            <StatList
              items={[
                ["Precision", asMetric(overview.model_metrics.precision)],
                ["Recall", asMetric(overview.model_metrics.recall)],
                ["F1 Score", asMetric(overview.model_metrics.f1)],
                ["PR-AUC", asMetric(overview.model_metrics.average_precision)],
                ["ROC-AUC", asMetric(overview.model_metrics.roc_auc)],
                ["Feature Count", String(overview.model_metrics.feature_count ?? "-")]
              ]}
            />
          </div>
          <div className="panel">
            <PanelHeader title="Priority Actions" subtitle="Bridges sorted for administrative review." />
            <PriorityTable bridges={overview.bridges} selectedBridgeId={selectedBridgeId} onSelect={onSelectBridge} />
          </div>
        </div>
      </section>
    </div>
  );
}

function BridgeOperationsView({
  detail,
  loading,
  selectedBridgeName
}: {
  detail: BridgeDetail | null;
  loading: boolean;
  selectedBridgeName: string;
}) {
  if (loading) {
    return <PanelPlaceholder label="Loading bridge operations workspace" tall />;
  }
  if (!detail) {
    return <PanelPlaceholder label={`Choose a bridge to open ${selectedBridgeName} operations`} tall />;
  }

  return (
    <div className="view-grid bridge-ops-view">
      <div className="panel bridge-twin-panel bridge-twin-panel--xl">
        <PanelHeader
          title={`${detail.bridge.bridge_name} Digital Twin`}
          subtitle="Bridge-specific 3D structural view with live anomaly pins, hover diagnostics, and stress hotspot overlays."
        />
        <BridgeTwin3D bridgeId={detail.bridge.bridge_id} bridgeName={detail.bridge.bridge_name} hotspots={detail.hotspots} anomalies={detail.anomalies} />
        <div className="bridge-twin-footer">
          <span>Drag to rotate</span>
          <span>Scroll to zoom</span>
          <span>Hover anomaly pins for model insight</span>
          <span>Hotspot orbs show stress concentration severity</span>
        </div>
      </div>

      <div className="content-grid bridge-ops-detail-grid">
        <div className="panel">
          <PanelHeader title="Bridge Posture" subtitle="Top-line bridge condition metrics for the current asset." />
          <div className="bridge-posture-grid">
            <InfoCard label="Peak Probability" value={formatPercent(detail.bridge.max_probability)} detail="Maximum anomaly probability on the selected bridge." compact />
            <InfoCard label="Anomaly Hits" value={String(detail.bridge.anomaly_count)} detail="Total rows classified as anomalous." compact />
            <InfoCard label="Mean SHI" value={formatDecimal(detail.bridge.mean_health_index)} detail="Average structural health index." compact />
            <InfoCard label="Top Hotspot" value={detail.bridge.top_hotspot} detail="Most common anomaly zone for the bridge." compact />
          </div>
        </div>

        <div className="panel">
          <PanelHeader title="Pipeline Runtime" subtitle="Inference stage trace for the active bridge analysis." />
          <Timeline stages={detail.runtime_trace} />
        </div>

        <div className="panel">
          <PanelHeader title="Stress Hotspots" subtitle="Locations on the bridge currently accumulating the highest stress signals." />
          <HotspotCards hotspots={detail.hotspots} />
        </div>
      </div>

      <div className="panel">
        <PanelHeader title="Output Verification" subtitle="Production-style evidence checks that validate whether the current model output is trustworthy." />
        <ValidationPanel validation={detail.validation} />
      </div>
    </div>
  );
}

function TelemetryView({
  detail,
  loading,
  telemetryMode,
  onTelemetryModeChange
}: {
  detail: BridgeDetail | null;
  loading: boolean;
  telemetryMode: TelemetryMode;
  onTelemetryModeChange: (mode: TelemetryMode) => void;
}) {
  if (loading) {
    return <PanelPlaceholder label="Loading telemetry lab" tall />;
  }
  if (!detail) {
    return <PanelPlaceholder label="Select a bridge to inspect telemetry" tall />;
  }

  return (
    <div className="content-grid telemetry-grid">
      <div className="panel telemetry-primary-panel">
        <PanelHeader title="Telemetry Channels" subtitle="Open one modality at a time for focused analysis." />
        <div className="mode-switcher">
          {(["gnss", "insar", "sensors"] as TelemetryMode[]).map((mode) => (
            <button
              key={mode}
              className={`mode-chip ${telemetryMode === mode ? "active" : ""}`}
              onClick={() => onTelemetryModeChange(mode)}
            >
              {mode.toUpperCase()}
            </button>
          ))}
        </div>
        <TelemetryPanel
          mode={telemetryMode}
          gnss={detail.telemetry.gnss}
          insar={detail.telemetry.insar}
          sensors={detail.telemetry.sensors}
        />
      </div>

      <div className="telemetry-secondary-stack">
        <div className="panel">
          <PanelHeader title="Explainability Drivers" subtitle="Counterfactual factors most responsible for the active bridge's anomaly profile." />
          <XaiList factors={detail.xai_factors} />
        </div>
        <div className="panel">
          <PanelHeader title="Anomaly Ledger" subtitle="Top-ranked anomalies for the active bridge." />
          <AnomalyTable rows={detail.anomalies} />
        </div>
      </div>
    </div>
  );
}

function InsarView({
  detail,
  loading,
  index,
  onIndexChange
}: {
  detail: BridgeDetail | null;
  loading: boolean;
  index: number;
  onIndexChange: (value: number) => void;
}) {
  if (loading) {
    return <PanelPlaceholder label="Loading InSAR explorer" tall />;
  }
  if (!detail) {
    return <PanelPlaceholder label="Choose a bridge to inspect InSAR frames" tall />;
  }
  return (
    <div className="content-grid insar-grid">
      <div className="panel">
        <PanelHeader title="InSAR Frame Explorer" subtitle="Frame timeline, heatmaps, interferograms, and segmented overlays." />
        <InsarViewer frames={detail.insar_frames} index={index} onIndexChange={onIndexChange} />
      </div>
      <div className="panel">
        <PanelHeader title="Operational Reading" subtitle="How the current frame should be interpreted by an administrator." />
        <FrameNarrative frame={detail.insar_frames[Math.min(index, Math.max(0, detail.insar_frames.length - 1))] ?? null} />
      </div>
    </div>
  );
}

function ReportsView({
  report,
  reports,
  selectedReport,
  onSelectReport,
  onGenerateReport,
  activeOperation
}: {
  report: ReportDetail | null;
  reports: OverviewResponse["reports"];
  selectedReport: string;
  onSelectReport: (name: string) => void;
  onGenerateReport: () => void;
  activeOperation: OperationStatus | null;
}) {
  return (
    <div className="view-grid reports-view">
      <div className="content-grid reports-grid">
        <div className="panel reports-control-panel">
          <PanelHeader title="CrewAI Intelligence Orchestrator" subtitle="One-click executive intelligence generation powered by Gemini 2.5 Pro and a multi-agent backend crew." />
          <div className="crew-feature-grid">
            <div className="crew-feature-card">
              <strong>Fleet Intelligence</strong>
              <span>Reviews all 6 bridges, model metrics, and multi-modal telemetry.</span>
            </div>
            <div className="crew-feature-card">
              <strong>Risk Triage</strong>
              <span>Ranks intervention priority, explains root causes, and builds the maintenance queue.</span>
            </div>
            <div className="crew-feature-card">
              <strong>Visual Strategy</strong>
              <span>Produces dashboard callouts, chart annotations, and operator prompts.</span>
            </div>
            <div className="crew-feature-card">
              <strong>Executive Briefing</strong>
              <span>Writes the archival engineering brief behind the analytical workspace.</span>
            </div>
          </div>
          <div className="reports-action-row">
            <button className="button-primary" onClick={onGenerateReport}>
              Run CrewAI Dashboard
            </button>
            <span>Gemini 2.5 Pro is loaded from the server `.env` configuration.</span>
          </div>
          {activeOperation && (
            <div className="crew-activity-inline">
              <strong>{activeOperation.message}</strong>
              <div className="progress-track">
                <div className="progress-bar" style={{ width: `${Math.max(6, activeOperation.progress)}%` }} />
              </div>
              <div className="crew-activity-steps">
                {activeOperation.steps.slice(-6).map((step, index) => (
                  <div key={`${step.stage}-${index}`} className="crew-activity-step">
                    <strong>{step.stage}</strong>
                    <span>{step.detail}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="panel">
          <PanelHeader title="Artifact Library" subtitle="Crew-generated dashboard runs and archived engineering briefings." />
          <div className="report-library">
            {reports.map((entry) => (
              <button
                key={entry.name}
                className={`report-library-item ${entry.name === selectedReport ? "active" : ""}`}
                onClick={() => onSelectReport(entry.name)}
              >
                <div>
                  <strong>{entry.title}</strong>
                  <small>{entry.kind === "crew_dashboard" ? "CrewAI dashboard artifact" : "Markdown report"}</small>
                </div>
                <span>{formatDate(entry.updated_at)}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="panel reports-dashboard-panel">
        <PanelHeader
          title={report?.dashboard?.hero.title || report?.name || "CrewAI Analytics Workspace"}
          subtitle="Admin analytical dashboard assembled from CrewAI agent outputs."
        />
        {report?.dashboard ? (
          <CrewDashboardView report={report} />
        ) : report ? (
          <pre className="markdown-viewer">{report.content}</pre>
        ) : (
          <PanelPlaceholder label="Run the CrewAI dashboard or select an archived artifact to inspect it here." tall />
        )}
      </div>
    </div>
  );
}

function CrewDashboardView({ report }: { report: ReportDetail }) {
  const dashboard = report.dashboard;
  if (!dashboard) {
    return <PanelPlaceholder label="CrewAI dashboard data is not available for this artifact." tall />;
  }

  return (
    <div className="crew-dashboard-shell">
      <section className="crew-hero">
        <div>
          <span className="kicker">CrewAI Executive Readout</span>
          <h3>{dashboard.hero.title}</h3>
          <p>{dashboard.hero.message}</p>
        </div>
        <div className="crew-watchlist">
          {dashboard.hero.watchlist.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      </section>

      <section className="crew-kpi-band">
        {dashboard.kpis.map((item) => (
          <InfoCard key={item.label} label={item.label} value={item.value} detail={item.detail} compact />
        ))}
      </section>

      <section className="crew-grid">
        <div className="panel">
          <PanelHeader title="Priority Actions" subtitle="Risk-ranked interventions emitted by the triage crew." />
          <div className="crew-priority-stack">
            {dashboard.priority_actions.map((item) => (
              <div key={`${item.bridge_id}-${item.recommendation}`} className="crew-priority-card">
                <div className="crew-priority-head">
                  <strong>{item.bridge_name}</strong>
                  <span className={`risk-pill ${item.risk_level.toLowerCase()}`}>{item.risk_level}</span>
                </div>
                <p>{item.recommendation}</p>
                <small>{item.rationale}</small>
                <div className="chip-row">
                  <span className="metric-chip-inline">Hotspot {item.dominant_hotspot}</span>
                  <span className="metric-chip-inline">Driver {item.primary_driver}</span>
                  <span className="metric-chip-inline">Probability {formatPercent(item.max_probability)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <PanelHeader title="Risk Distribution" subtitle="Crew-rated bridge severity mix and recommendation volume." />
          <CategoryBars title="Risk Levels" data={dashboard.risk_distribution} />
          <CategoryBars title="Recommendations" data={dashboard.recommendation_distribution} />
        </div>
        <div className="panel">
          <PanelHeader title="Fleet Risk Ladder" subtitle="All six bridges ranked by anomaly probability with proper bridge naming." />
          <BridgeRiskBars data={dashboard.bridge_risk_series} />
        </div>
      </section>

      <section className="crew-grid">
        <div className="panel">
          <PanelHeader title="Driver Breakdown" subtitle="Dominant driver and urgency score for each bridge in the intervention queue." />
          <DriverBars data={dashboard.driver_breakdown} />
        </div>
        <div className="panel">
          <PanelHeader title="Agent Intelligence" subtitle="What each CrewAI specialist contributed to the admin workspace." />
          <div className="agent-panel-stack">
            {dashboard.agent_panels.map((panel) => (
              <div key={panel.agent} className="agent-panel-card">
                <span className="kicker">{panel.agent}</span>
                <strong>{panel.headline}</strong>
                <p>{panel.body}</p>
                <div className="agent-bullet-stack">
                  {panel.bullets.map((bullet) => (
                    <span key={bullet}>{bullet}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="crew-grid">
        <div className="panel">
          <PanelHeader title="Hotspot Mix" subtitle="Which structural zones dominate the crew intervention queue." />
          <CategoryBars title="Dominant Hotspots" data={dashboard.hotspot_mix} />
        </div>
        <div className="panel">
          <PanelHeader title="Chart Annotations" subtitle="Crew-authored chart-side messages for operators and leadership." />
          <NarrativeList items={dashboard.chart_annotations} />
        </div>
        <div className="panel">
          <PanelHeader title="Operator Prompts" subtitle="Recommended admin follow-ups generated by the dashboard strategist." />
          <NarrativeList items={dashboard.operator_prompts} />
        </div>
        <div className="panel">
          <PanelHeader title="Maintenance Queue" subtitle="Action queue synthesized by the triage engineer." />
          <NarrativeList items={dashboard.maintenance_queue} />
        </div>
      </section>

      <section className="panel">
        <PanelHeader title="Archival Brief" subtitle="Crew-authored markdown briefing preserved alongside the dashboard artifact." />
        <pre className="markdown-viewer">{dashboard.markdown}</pre>
      </section>
    </div>
  );
}

function BridgeRiskBars({
  data
}: {
  data: Array<{ bridge_name: string; max_probability: number | null; anomaly_count: number; top_hotspot: string }>;
}) {
  const max = Math.max(...data.map((item) => item.max_probability ?? 0), 0.01);
  return (
    <div className="bridge-risk-bars">
      {data.map((item) => (
        <div key={item.bridge_name} className="driver-bar-card">
          <div className="driver-bar-head">
            <div>
              <strong>{item.bridge_name}</strong>
              <small>{item.top_hotspot} hotspot</small>
            </div>
            <span className="metric-chip-inline">{item.anomaly_count} hits</span>
          </div>
          <div className="category-bar-track">
            <div className="category-bar-fill" style={{ width: `${((item.max_probability ?? 0) / max) * 100}%` }} />
          </div>
          <small>Peak anomaly probability {formatPercent(item.max_probability)}</small>
        </div>
      ))}
    </div>
  );
}

function CategoryBars({
  title,
  data
}: {
  title: string;
  data: Array<{ label: string; count: number }>;
}) {
  const max = Math.max(...data.map((item) => item.count), 1);
  return (
    <div className="category-bars">
      <strong>{title}</strong>
      {data.map((item) => (
        <div key={item.label} className="category-bar-row">
          <span>{item.label}</span>
          <div className="category-bar-track">
            <div className="category-bar-fill" style={{ width: `${(item.count / max) * 100}%` }} />
          </div>
          <strong>{item.count}</strong>
        </div>
      ))}
    </div>
  );
}

function DriverBars({
  data
}: {
  data: Array<{ bridge_name: string; driver: string; urgency_score: number; risk_level: string }>;
}) {
  const max = Math.max(...data.map((item) => item.urgency_score), 1);
  return (
    <div className="driver-bars">
      {data.map((item) => (
        <div key={`${item.bridge_name}-${item.driver}`} className="driver-bar-card">
          <div className="driver-bar-head">
            <div>
              <strong>{item.bridge_name}</strong>
              <small>{item.driver}</small>
            </div>
            <span className={`risk-pill ${item.risk_level.toLowerCase()}`}>{item.risk_level}</span>
          </div>
          <div className="category-bar-track">
            <div className="category-bar-fill" style={{ width: `${(item.urgency_score / max) * 100}%` }} />
          </div>
          <small>Urgency score {item.urgency_score}</small>
        </div>
      ))}
    </div>
  );
}

function NarrativeList({ items }: { items: string[] }) {
  if (items.length === 0) {
    return <PanelPlaceholder label="No crew narrative available for this section yet." />;
  }
  return (
    <div className="narrative-list">
      {items.map((item) => (
        <div key={item} className="narrative-list-item">
          {item}
        </div>
      ))}
    </div>
  );
}

function BridgeTwin3D({
  bridgeId,
  bridgeName,
  hotspots,
  anomalies
}: {
  bridgeId: string;
  bridgeName: string;
  hotspots: Hotspot[];
  anomalies: AnomalyRow[];
}) {
  const config = useMemo(() => bridgeSceneConfig(bridgeId), [bridgeId]);
  const anomalyMarkers = useMemo(() => buildAnomalyMarkers(anomalies, bridgeId, config), [anomalies, bridgeId, config]);
  const hotspotMarkers = useMemo(() => buildStressMarkers(hotspots, config), [hotspots, config]);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const hoveredMarker = anomalyMarkers.find((marker) => marker.id === hoveredId) ?? null;

  return (
    <div className="canvas-shell canvas-shell--bridge">
      <div className="bridge-canvas-overlay">
        <div className="bridge-overlay-chip">
          <strong>{bridgeName}</strong>
          <span>{humanize(config.cableMode)} bridge profile</span>
        </div>
        {hoveredMarker ? (
          <div className="bridge-insight-card">
            <span className="kicker">Hovered Anomaly</span>
            <strong>{hoveredMarker.title}</strong>
            <p>{hoveredMarker.detail}</p>
          </div>
        ) : (
          <div className="bridge-insight-card muted">
            <span className="kicker">Twin Guidance</span>
            <strong>Inspect anomaly pins</strong>
            <p>Move over the red pins on the bridge to read the exact anomaly timestamp, severity, and operating condition.</p>
          </div>
        )}
      </div>

      <Canvas shadows camera={{ position: config.cameraPosition, fov: config.cameraFov }}>
        <color attach="background" args={["#f4eee4"]} />
        <fog attach="fog" args={["#f4eee4", 22, 52]} />
        <ambientLight intensity={1.2} />
        <hemisphereLight args={["#fff6e9", "#c8b59d", 1.05]} />
        <directionalLight castShadow position={[18, 18, 12]} intensity={2.1} shadow-mapSize-width={2048} shadow-mapSize-height={2048} />

        <group position={config.groupPosition} rotation={config.groupRotation} scale={1}>
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2.2, 0]} receiveShadow>
            <planeGeometry args={[48, 30]} />
            <meshStandardMaterial color="#d8c9b6" roughness={0.98} />
          </mesh>

          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.95, -4.4]} receiveShadow>
            <planeGeometry args={[46, 7.6]} />
            <meshStandardMaterial color="#b3c5d1" roughness={0.24} metalness={0.05} />
          </mesh>

          <RoundedBox
            args={[config.deckLength, config.deckThickness, config.deckWidth]}
            radius={0.08}
            smoothness={6}
            position={[0, config.deckY, 0]}
            castShadow
            receiveShadow
          >
            <meshStandardMaterial color="#5b5651" metalness={0.35} roughness={0.58} />
          </RoundedBox>

          <RoundedBox
            args={[config.deckLength * 0.32, 0.18, config.deckWidth * 0.56]}
            radius={0.04}
            smoothness={4}
            position={[0, config.deckY + 0.35, 0]}
            castShadow
          >
            <meshStandardMaterial color="#cfb595" metalness={0.18} roughness={0.62} />
          </RoundedBox>

          {config.piers.map((pier) => (
            <RoundedBox
              key={`pier-${pier.x}`}
              args={[pier.width, pier.height, Math.max(1.2, config.deckWidth * 0.38)]}
              radius={0.05}
              position={[pier.x, config.deckY - pier.height / 2 - 0.28, 0]}
              castShadow
              receiveShadow
            >
              <meshStandardMaterial color="#8f7964" roughness={0.78} />
            </RoundedBox>
          ))}

          {config.towers.map((tower) => (
            <group key={`tower-${tower.x}`} position={[tower.x, config.deckY + tower.height / 2 + 0.2, 0]}>
              <RoundedBox args={[tower.width, tower.height, 1.18]} radius={0.06} position={[0, 0, 0]} castShadow receiveShadow>
                <meshStandardMaterial color="#7a6a5b" metalness={0.2} roughness={0.65} />
              </RoundedBox>
              <RoundedBox args={[tower.width * 1.85, 0.3, 1.28]} radius={0.05} position={[0, tower.height / 2 - 0.12, 0]} castShadow receiveShadow>
                <meshStandardMaterial color="#6a5d50" metalness={0.22} roughness={0.56} />
              </RoundedBox>
            </group>
          ))}

          {renderBridgeCables(config)}

          {config.cableMode === "hybrid" && (
            <mesh position={[0, config.deckY + 3.85, 0]} castShadow receiveShadow>
              <torusGeometry args={[config.deckLength * 0.22, 0.1, 16, 64, Math.PI]} />
              <meshStandardMaterial color="#b49f89" metalness={0.35} roughness={0.45} />
            </mesh>
          )}

          <mesh position={[0, config.deckY - 0.32, config.deckWidth / 2 - 0.08]} castShadow receiveShadow>
            <boxGeometry args={[config.deckLength + 0.2, 0.12, 0.08]} />
            <meshStandardMaterial color="#d6bf9f" metalness={0.18} roughness={0.62} />
          </mesh>
          <mesh position={[0, config.deckY - 0.32, -config.deckWidth / 2 + 0.08]} castShadow receiveShadow>
            <boxGeometry args={[config.deckLength + 0.2, 0.12, 0.08]} />
            <meshStandardMaterial color="#d6bf9f" metalness={0.18} roughness={0.62} />
          </mesh>

          {hotspotMarkers.map((marker) => (
            <StressMarker key={marker.id} marker={marker} />
          ))}
          {anomalyMarkers.map((marker) => (
            <AnomalyPin
              key={marker.id}
              marker={marker}
              active={hoveredId === marker.id}
              onHover={() => setHoveredId(marker.id)}
              onLeave={() => setHoveredId((current) => (current === marker.id ? null : current))}
            />
          ))}
        </group>

        <OrbitControls
          enablePan={false}
          enableDamping
          dampingFactor={0.08}
          target={config.cameraTarget}
          minPolarAngle={0.82}
          maxPolarAngle={Math.PI / 2.02}
          minDistance={config.minDistance}
          maxDistance={config.maxDistance}
        />
      </Canvas>
    </div>
  );
}

function OperationPanel({
  operation,
  minimized,
  onToggleMinimize
}: {
  operation: OperationStatus;
  minimized: boolean;
  onToggleMinimize: () => void;
}) {
  return (
    <section className="panel operation-panel">
      <div className="operation-panel-header">
        <PanelHeader
          title={operation.kind === "full_pipeline" ? "Backend Pipeline Activity" : "Bridge Refresh Activity"}
          subtitle="Live backend progress for the current action."
        />
        <button className="minimize-button" onClick={onToggleMinimize}>
          {minimized ? "Expand" : "Minimize"}
        </button>
      </div>
      <div className="operation-head">
        <div>
          <strong>{operation.message}</strong>
          <small>
            {humanize(operation.kind)}{operation.target ? ` · ${operation.target}` : ""}
          </small>
        </div>
        <div className={`operation-status ${operation.status}`}>{operation.status}</div>
      </div>
      <div className="progress-track">
        <div className="progress-bar" style={{ width: `${Math.max(6, operation.progress)}%` }} />
      </div>
      {!minimized && (
        <div className="operation-steps">
          {operation.steps.length === 0 && <PanelPlaceholder label="Waiting for backend stages..." />}
          {operation.steps.map((step, index) => (
            <div key={`${step.stage}-${index}`} className="operation-step">
              <div className="operation-step-title">
                <strong>{step.stage}</strong>
                <span>{step.status}</span>
              </div>
              <p>{step.detail}</p>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function renderBridgeCables(config: BridgeSceneConfig) {
  const cableColor = "#bca58b";
  const depth = config.deckWidth * 0.36;
  const deckAnchorY = config.deckY + config.deckThickness * 0.38;

  if (config.cableMode === "suspension") {
    const towerLeft = config.towers[0];
    const towerRight = config.towers[config.towers.length - 1];
    const anchors = Array.from({ length: 12 }, (_, index) => -config.deckLength / 2 + 1.3 + index * ((config.deckLength - 2.6) / 11));

    return (
      <>
        {anchors.flatMap((x) => {
          const sourceTower = x < 0 ? towerLeft : towerRight;
          const towerTopY = config.deckY + sourceTower.height + 0.18;

          return [depth, -depth].map((z) => (
            <BridgeCable
              key={`${x}-${z}`}
              start={[sourceTower.x, towerTopY, z]}
              end={[x, deckAnchorY, z]}
              color={cableColor}
            />
          ));
        })}
      </>
    );
  }

  const fanAnchors = config.towers.flatMap((tower) =>
    Array.from({ length: 5 }, (_, index) => {
      const spread = index - 2;
      const deckX = tower.x + spread * (config.deckLength * 0.11);
      const towerTopY = config.deckY + tower.height + 0.16;
      const towerX = tower.x + spread * 0.18;
      const towerY = towerTopY - Math.abs(spread) * 0.12;
      return { deckX, towerX, towerY };
    })
  );

  return (
    <>
      {fanAnchors.flatMap((anchor) =>
        [depth, -depth].map((z) => (
          <BridgeCable
            key={`${anchor.deckX}-${anchor.towerX}-${z}`}
            start={[anchor.towerX, anchor.towerY, z]}
            end={[anchor.deckX, deckAnchorY, z]}
            color={cableColor}
          />
        ))
      )}
    </>
  );
}

function BridgeCable({
  start,
  end,
  color
}: {
  start: [number, number, number];
  end: [number, number, number];
  color: string;
}) {
  const segment = useMemo(() => {
    const startVector = new THREE.Vector3(...start);
    const endVector = new THREE.Vector3(...end);
    const midpoint = startVector.clone().add(endVector).multiplyScalar(0.5);
    const direction = endVector.clone().sub(startVector);
    const length = direction.length();
    const quaternion = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      direction.clone().normalize()
    );

    return {
      position: midpoint.toArray() as [number, number, number],
      quaternion,
      length
    };
  }, [start, end]);

  return (
    <mesh position={segment.position} quaternion={segment.quaternion} castShadow>
      <cylinderGeometry args={[0.04, 0.04, segment.length, 14]} />
      <meshStandardMaterial color={color} metalness={0.5} roughness={0.28} />
    </mesh>
  );
}

function StressMarker({ marker }: { marker: PositionedHotspot }) {
  const { hotspot, position } = marker;
  const scale = 0.52 + Math.min(hotspot.hit_count, 12) * 0.05;
  const color =
    hotspot.max_probability && hotspot.max_probability >= 0.95
      ? "#c75a36"
      : hotspot.max_probability && hotspot.max_probability >= 0.7
        ? "#e39a45"
        : "#4f8763";

  return (
    <group position={position}>
      <mesh scale={scale} castShadow>
        <sphereGeometry args={[0.26, 24, 24]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.45} />
      </mesh>
      <mesh position={[0, -0.36, 0]}>
        <cylinderGeometry args={[0.03, 0.03, 0.45, 10]} />
        <meshStandardMaterial color="#6a5d50" roughness={0.75} />
      </mesh>
      <Html distanceFactor={18} position={[0, 0.58, 0]} center>
        <div className="bridge-marker-label">{hotspot.zone}</div>
      </Html>
    </group>
  );
}

function AnomalyPin({
  marker,
  active,
  onHover,
  onLeave
}: {
  marker: PositionedAnomaly;
  active: boolean;
  onHover: () => void;
  onLeave: () => void;
}) {
  const color = marker.severity >= 0.95 ? "#b53a24" : marker.severity >= 0.8 ? "#d97432" : "#d9a43b";

  return (
    <group position={marker.position} onPointerOver={(event) => {
      event.stopPropagation();
      onHover();
    }} onPointerOut={(event) => {
      event.stopPropagation();
      onLeave();
    }}>
      <mesh castShadow scale={active ? 1.24 : 1}>
        <sphereGeometry args={[0.15, 18, 18]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={active ? 0.82 : 0.55} />
      </mesh>
      <mesh position={[0, -0.32, 0]} castShadow>
        <cylinderGeometry args={[0.02, 0.02, 0.38, 8]} />
        <meshStandardMaterial color="#4b3f33" roughness={0.8} />
      </mesh>
      {active && (
        <Html distanceFactor={18} position={[0, 0.5, 0]} center>
          <div className="bridge-pin-badge">{marker.title}</div>
        </Html>
      )}
    </group>
  );
}

function buildStressMarkers(hotspots: Hotspot[], config: BridgeSceneConfig): PositionedHotspot[] {
  return hotspots.map((hotspot, index) => {
    const zone = normalizeZone(hotspot.zone);
    const anchors = config.zoneAnchors[zone] || config.zoneAnchors.deck;
    const anchor = anchors[index % anchors.length];
    const xShift = ((hotspot.x ?? 50) - 50) / 100 * 1.8;
    const zShift = ((hotspot.y ?? 50) - 50) / 100 * 0.75;

    return {
      id: `${zone}-${index}`,
      hotspot,
      position: [anchor[0] + xShift, anchor[1] + 0.18, anchor[2] + zShift]
    };
  });
}

function buildAnomalyMarkers(rows: AnomalyRow[], bridgeId: string, config: BridgeSceneConfig): PositionedAnomaly[] {
  return rows.slice(0, 14).map((row, index) => {
    const zone = normalizeZone(row.hotspot);
    const anchors = config.zoneAnchors[zone] || config.zoneAnchors.deck;
    const seed = stableHash(`${bridgeId}-${row.timestamp ?? "na"}-${row.hotspot}-${index}`);
    const anchor = anchors[seed % anchors.length];
    const xOffset = (((seed % 9) - 4) / 10) * 0.34;
    const zOffset = (((Math.floor(seed / 7) % 7) - 3) / 10) * 0.24;
    const elevation = 0.22 + (row.anomaly_probability ?? 0) * 0.26;

    return {
      id: `${bridgeId}-${row.timestamp ?? index}-${index}`,
      severity: row.anomaly_probability ?? 0,
      position: [anchor[0] + xOffset, anchor[1] + elevation, anchor[2] + zOffset],
      row,
      title: `${humanize(zone)} anomaly · ${formatPercent(row.anomaly_probability ?? 0)}`,
      detail: [
        row.timestamp ? formatDate(row.timestamp) : "Timestamp unavailable",
        row.deflection_mm != null ? `Deflection ${formatDecimal(row.deflection_mm)} mm` : null,
        row.vibration_ms2 != null ? `Vibration ${formatDecimal(row.vibration_ms2)} m/s²` : null,
        row.failure_probability != null ? `PoF ${formatPercent(row.failure_probability)}` : null,
        row.health_index != null ? `SHI ${formatDecimal(row.health_index)}` : null
      ]
        .filter(Boolean)
        .join(" · ")
    };
  });
}

function bridgeSceneConfig(bridgeId: string): BridgeSceneConfig {
  const configs: Record<string, BridgeSceneConfig> = {
    bridge_alpha: {
      deckLength: 19.5,
      deckWidth: 3.6,
      deckThickness: 0.6,
      deckY: 2.2,
      groupPosition: [0, -0.9, 0],
      groupRotation: [-0.08, 0.22, 0],
      cameraPosition: [0, 8.8, 22.5],
      cameraFov: 30,
      cameraTarget: [0, 3.8, 0],
      minDistance: 15,
      maxDistance: 29,
      piers: [
        { x: -7.6, height: 3.6, width: 1.08 },
        { x: -3.6, height: 3.9, width: 1.02 },
        { x: 0, height: 4.1, width: 1.15 },
        { x: 3.7, height: 3.75, width: 1.02 },
        { x: 7.8, height: 3.55, width: 1.08 }
      ],
      towers: [
        { x: -5.8, height: 7.4, width: 1.2 },
        { x: 5.9, height: 7.1, width: 1.16 }
      ],
      cableMode: "suspension",
      zoneAnchors: {
        deck: [[-4.4, 2.56, 0], [0.2, 2.56, 0], [4.8, 2.56, 0]],
        tower: [[-5.8, 8.6, 0], [5.9, 8.25, 0]],
        cable: [[-2.7, 6.65, 1.05], [2.8, 6.45, -1.05], [5.0, 5.8, 1.05]],
        pier: [[-7.6, 0.62, 0], [0, 0.55, 0], [7.8, 0.64, 0]],
        joint: [[-8.8, 2.35, 0], [8.7, 2.35, 0]]
      }
    },
    bridge_beta: {
      deckLength: 18.4,
      deckWidth: 3.3,
      deckThickness: 0.54,
      deckY: 2.15,
      groupPosition: [0, -0.95, 0],
      groupRotation: [-0.06, -0.06, 0],
      cameraPosition: [0, 9.3, 21],
      cameraFov: 28,
      cameraTarget: [0, 3.4, 0],
      minDistance: 14,
      maxDistance: 27,
      piers: [
        { x: -6.7, height: 4.2, width: 0.98 },
        { x: -2.4, height: 4.45, width: 0.94 },
        { x: 2.3, height: 4.32, width: 0.94 },
        { x: 6.6, height: 4.05, width: 0.98 }
      ],
      towers: [
        { x: -3.4, height: 6.7, width: 1.1 },
        { x: 3.4, height: 6.7, width: 1.1 }
      ],
      cableMode: "fan",
      zoneAnchors: {
        deck: [[-5.5, 2.5, 0], [-0.5, 2.5, 0], [4.9, 2.5, 0]],
        tower: [[-3.4, 7.9, 0], [3.4, 7.9, 0]],
        cable: [[-4.8, 5.8, 0.95], [-1.4, 6.3, -0.95], [4.8, 5.8, 0.95]],
        pier: [[-6.7, 0.38, 0], [2.3, 0.38, 0], [6.6, 0.4, 0]],
        joint: [[-8.2, 2.28, 0], [8.2, 2.28, 0]]
      }
    },
    bridge_gamma: {
      deckLength: 20.2,
      deckWidth: 3.5,
      deckThickness: 0.58,
      deckY: 2.05,
      groupPosition: [0, -0.92, 0],
      groupRotation: [-0.07, 0.14, 0],
      cameraPosition: [0, 9.1, 23],
      cameraFov: 30,
      cameraTarget: [0, 3.55, 0],
      minDistance: 15,
      maxDistance: 29,
      piers: [
        { x: -7.3, height: 3.5, width: 1.0 },
        { x: -3.0, height: 3.9, width: 0.95 },
        { x: 0, height: 4.4, width: 1.15 },
        { x: 3.0, height: 3.9, width: 0.95 },
        { x: 7.3, height: 3.5, width: 1.0 }
      ],
      towers: [
        { x: -5.0, height: 6.5, width: 1.05 },
        { x: 0, height: 7.2, width: 1.2 },
        { x: 5.0, height: 6.5, width: 1.05 }
      ],
      cableMode: "hybrid",
      zoneAnchors: {
        deck: [[-5.8, 2.38, 0], [0, 2.38, 0], [5.8, 2.38, 0]],
        tower: [[-5.0, 7.35, 0], [0, 8.2, 0], [5.0, 7.35, 0]],
        cable: [[-2.8, 5.8, 1.02], [0, 6.7, -1.02], [2.8, 5.8, 1.02]],
        pier: [[-7.3, 0.48, 0], [0, -0.04, 0], [7.3, 0.48, 0]],
        joint: [[-9.4, 2.15, 0], [9.4, 2.15, 0]]
      }
    },
    bridge_delta: {
      deckLength: 17.2,
      deckWidth: 3.2,
      deckThickness: 0.54,
      deckY: 2.08,
      groupPosition: [0, -0.96, 0],
      groupRotation: [-0.05, -0.18, 0],
      cameraPosition: [0, 8.9, 20.8],
      cameraFov: 28,
      cameraTarget: [0, 3.45, 0],
      minDistance: 14,
      maxDistance: 26,
      piers: [
        { x: -5.8, height: 4.05, width: 1.0 },
        { x: -1.9, height: 4.25, width: 0.92 },
        { x: 2.0, height: 4.25, width: 0.92 },
        { x: 5.8, height: 4.05, width: 1.0 }
      ],
      towers: [
        { x: -4.0, height: 6.1, width: 1.08 },
        { x: 4.1, height: 6.1, width: 1.08 }
      ],
      cableMode: "fan",
      zoneAnchors: {
        deck: [[-4.9, 2.42, 0], [0, 2.42, 0], [4.9, 2.42, 0]],
        tower: [[-4.0, 7.12, 0], [4.1, 7.12, 0]],
        cable: [[-5.4, 5.58, 0.95], [-1.7, 5.95, -0.95], [5.4, 5.58, 0.95]],
        pier: [[-5.8, 0.1, 0], [2.0, 0.1, 0], [5.8, 0.1, 0]],
        joint: [[-7.6, 2.2, 0], [7.6, 2.2, 0]]
      }
    },
    bridge_epsilon: {
      deckLength: 21.4,
      deckWidth: 3.8,
      deckThickness: 0.62,
      deckY: 2.16,
      groupPosition: [0, -0.9, 0],
      groupRotation: [-0.08, 0.28, 0],
      cameraPosition: [0, 9.2, 24],
      cameraFov: 30,
      cameraTarget: [0, 3.6, 0],
      minDistance: 16,
      maxDistance: 30,
      piers: [
        { x: -8.3, height: 3.5, width: 1.08 },
        { x: -4.4, height: 3.8, width: 1.02 },
        { x: 0, height: 4.05, width: 1.12 },
        { x: 4.4, height: 3.8, width: 1.02 },
        { x: 8.3, height: 3.5, width: 1.08 }
      ],
      towers: [
        { x: -6.4, height: 7.15, width: 1.16 },
        { x: 6.4, height: 7.15, width: 1.16 }
      ],
      cableMode: "suspension",
      zoneAnchors: {
        deck: [[-6.4, 2.53, 0], [0, 2.53, 0], [6.4, 2.53, 0]],
        tower: [[-6.4, 8.34, 0], [6.4, 8.34, 0]],
        cable: [[-3.9, 6.42, 1.08], [0, 5.84, -1.08], [3.9, 6.42, 1.08]],
        pier: [[-8.3, 0.57, 0], [0, 0.3, 0], [8.3, 0.57, 0]],
        joint: [[-10.1, 2.32, 0], [10.1, 2.32, 0]]
      }
    },
    bridge_zeta: {
      deckLength: 18.9,
      deckWidth: 3.45,
      deckThickness: 0.57,
      deckY: 2.12,
      groupPosition: [0, -0.94, 0],
      groupRotation: [-0.07, -0.12, 0],
      cameraPosition: [0, 9.0, 22],
      cameraFov: 29,
      cameraTarget: [0, 3.55, 0],
      minDistance: 15,
      maxDistance: 28,
      piers: [
        { x: -6.9, height: 3.9, width: 1.0 },
        { x: -2.3, height: 4.2, width: 0.94 },
        { x: 2.2, height: 4.2, width: 0.94 },
        { x: 6.8, height: 3.9, width: 1.0 }
      ],
      towers: [
        { x: -4.7, height: 6.9, width: 1.12 },
        { x: 4.7, height: 6.9, width: 1.12 }
      ],
      cableMode: "hybrid",
      zoneAnchors: {
        deck: [[-5.5, 2.47, 0], [0, 2.47, 0], [5.5, 2.47, 0]],
        tower: [[-4.7, 8.02, 0], [4.7, 8.02, 0]],
        cable: [[-2.6, 6.18, 1.0], [0, 6.7, -1.0], [2.6, 6.18, 1.0]],
        pier: [[-6.9, 0.22, 0], [2.2, 0.22, 0], [6.8, 0.22, 0]],
        joint: [[-8.8, 2.24, 0], [8.8, 2.24, 0]]
      }
    }
  };

  return configs[bridgeId] ?? configs.bridge_alpha;
}

function normalizeZone(zone: string | null | undefined): string {
  const key = String(zone ?? "Deck").toLowerCase();
  if (key.includes("tower")) return "tower";
  if (key.includes("cable")) return "cable";
  if (key.includes("pier")) return "pier";
  if (key.includes("joint")) return "joint";
  return "deck";
}

function stableHash(input: string): number {
  let hash = 0;
  for (let index = 0; index < input.length; index += 1) {
    hash = (hash * 31 + input.charCodeAt(index)) >>> 0;
  }
  return hash;
}

function FleetMap({
  bridges,
  selectedBridgeId,
  onSelect
}: {
  bridges: BridgeCard[];
  selectedBridgeId: string;
  onSelect: (bridgeId: string) => void;
}) {
  const projected = bridges.map((bridge) => ({
    ...bridge,
    x: ((bridge.lon + 125) / 60) * 100,
    y: (1 - (bridge.lat - 24) / 25) * 100
  }));
  const selected = projected.find((bridge) => bridge.bridge_id === selectedBridgeId) ?? projected[0];

  return (
    <div className="map-shell">
      <svg viewBox="0 0 100 100" className="fleet-map">
        <rect x="1" y="1" width="98" height="98" rx="12" fill="#f4eee4" stroke="rgba(66,54,43,0.12)" />
        {selected &&
          projected
            .filter((bridge) => bridge.bridge_id !== selected.bridge_id)
            .map((bridge) => (
              <path
                key={`${selected.bridge_id}-${bridge.bridge_id}`}
                d={`M ${selected.x} ${selected.y} Q ${(selected.x + bridge.x) / 2} ${Math.min(selected.y, bridge.y) - 10} ${bridge.x} ${bridge.y}`}
                stroke="rgba(114, 92, 67, 0.18)"
                strokeWidth="0.8"
                fill="none"
              />
            ))}
        {projected.map((bridge) => (
          <g key={bridge.bridge_id} onClick={() => onSelect(bridge.bridge_id)} className="map-node">
            <circle
              cx={bridge.x}
              cy={bridge.y}
              r={bridge.bridge_id === selectedBridgeId ? 3.6 : 2.8}
              fill={bridge.bridge_id === selectedBridgeId ? "#2f5d50" : riskColor(bridge.max_probability)}
            />
            <text x={bridge.x + 2.5} y={bridge.y - 2.5}>
              {bridge.bridge_name}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function PriorityTable({
  bridges,
  selectedBridgeId,
  onSelect
}: {
  bridges: BridgeCard[];
  selectedBridgeId: string;
  onSelect: (bridgeId: string) => void;
}) {
  return (
    <div className="priority-list">
      {bridges.slice(0, 6).map((bridge, index) => (
        <button
          key={bridge.bridge_id}
          className={`priority-item ${bridge.bridge_id === selectedBridgeId ? "active" : ""}`}
          onClick={() => onSelect(bridge.bridge_id)}
        >
          <span className="priority-rank">{index + 1}</span>
          <div>
            <strong>{bridge.bridge_name}</strong>
            <small>
              {bridge.city} · {bridge.top_hotspot}
            </small>
          </div>
          <em>{formatPercent(bridge.max_probability)}</em>
        </button>
      ))}
    </div>
  );
}

function TelemetryPanel({
  mode,
  gnss,
  insar,
  sensors
}: {
  mode: TelemetryMode;
  gnss: TimePoint[];
  insar: TimePoint[];
  sensors: TimePoint[];
}) {
  if (mode === "gnss") {
    return <LineChartPanel title="GNSS Displacement Trend" data={gnss} series={[{ key: "total_mm", label: "Displacement", color: "#2f5d50" }]} />;
  }
  if (mode === "insar") {
    return <LineChartPanel title="InSAR LOS Deformation" data={insar} series={[{ key: "los_displacement", label: "LOS displacement", color: "#bb6f39" }]} />;
  }
  return (
    <LineChartPanel
      title="Sensor Fusion Panel"
      data={sensors}
      series={[
        { key: "Deflection_mm", label: "Deflection", color: "#745c43" },
        { key: "Vibration_ms2", label: "Vibration", color: "#c75a36" },
        { key: "Probability_of_Failure_PoF", label: "PoF", color: "#2f5d50" },
        { key: "Structural_Health_Index_SHI", label: "SHI", color: "#8f7964" }
      ]}
    />
  );
}

function LineChartPanel({
  title,
  data,
  series
}: {
  title: string;
  data: TimePoint[];
  series: Array<{ key: string; label: string; color: string }>;
}) {
  if (data.length === 0) {
    return <PanelPlaceholder label="No telemetry available" />;
  }

  const width = 900;
  const height = 320;
  const padding = 28;
  const values = series.flatMap((item) => data.map((row) => Number(row[item.key]))).filter((value) => Number.isFinite(value));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const pathFor = (key: string) =>
    data
      .map((row, index) => {
        const value = Number(row[key]);
        if (!Number.isFinite(value)) {
          return null;
        }
        const x = padding + (index / Math.max(1, data.length - 1)) * (width - padding * 2);
        const y = height - padding - ((value - min) / range) * (height - padding * 2);
        return `${index === 0 ? "M" : "L"} ${x} ${y}`;
      })
      .filter(Boolean)
      .join(" ");

  return (
    <div className="chart-wrap">
      <div className="chart-toolbar">
        <strong>{title}</strong>
        <div className="chart-legend">
          {series.map((item) => (
            <span key={item.key}>
              <i style={{ background: item.color }} />
              {item.label}
            </span>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="telemetry-chart">
        {[0, 1, 2, 3].map((tick) => {
          const y = padding + (tick / 3) * (height - padding * 2);
          return <line key={tick} x1={padding} y1={y} x2={width - padding} y2={y} />;
        })}
        {series.map((item) => (
          <path key={item.key} d={pathFor(item.key)} stroke={item.color} />
        ))}
      </svg>
    </div>
  );
}

function InsarViewer({
  frames,
  index,
  onIndexChange
}: {
  frames: InSarFrame[];
  index: number;
  onIndexChange: (value: number) => void;
}) {
  if (frames.length === 0) {
    return <PanelPlaceholder label="No InSAR frames are available" />;
  }
  const frame = frames[Math.min(index, frames.length - 1)];
  return (
    <div className="insar-shell">
      <div className="insar-toolbar">
        <span>Frame {index + 1}</span>
        <input type="range" min={0} max={frames.length - 1} value={Math.min(index, frames.length - 1)} onChange={(event) => onIndexChange(Number(event.target.value))} />
        <strong>{frame.timestamp ? formatDate(frame.timestamp) : "No timestamp"}</strong>
      </div>
      <div className="chip-row">
        <MetricChip label="Mask Ratio" value={formatDecimal(frame.mask_ratio)} />
        <MetricChip label="Energy" value={formatDecimal(frame.deformation_energy)} />
        <MetricChip label="Coherence" value={formatDecimal(frame.coherence_mean)} />
      </div>
      <div className="image-grid">
        <ImageCard title="Amplitude" src={frame.image_path} />
        <ImageCard title="Interferogram" src={frame.interferogram_path} />
        <ImageCard title="Heatmap" src={frame.heatmap_path} />
        <ImageCard title="Coherence" src={frame.coherence_path} />
      </div>
      <ImageCard title="Segmented Overlay" src={frame.overlay_path} wide />
    </div>
  );
}

function FrameNarrative({ frame }: { frame: InSarFrame | null }) {
  if (!frame) {
    return <PanelPlaceholder label="No frame selected" />;
  }
  const stressLevel =
    (frame.mask_ratio ?? 0) > 0.02 || (frame.deformation_energy ?? 0) > 0.2
      ? "Elevated deformation response"
      : "Stable deformation response";

  return (
    <div className="narrative-card">
      <strong>{stressLevel}</strong>
      <p>
        This frame shows a mask ratio of {formatDecimal(frame.mask_ratio)} and a deformation energy of{" "}
        {formatDecimal(frame.deformation_energy)}. Higher values indicate a wider or more intense deformation pattern.
      </p>
      <p>
        The coherence score is {formatDecimal(frame.coherence_mean)}, which helps indicate whether the frame remains
        reliable enough for administrative review and downstream engineering investigation.
      </p>
    </div>
  );
}

function Timeline({ stages }: { stages: RuntimeStage[] }) {
  return (
    <div className="timeline">
      {stages.map((stage) => (
        <div key={stage.stage} className="timeline-entry">
          <div className="timeline-dot" />
          <div>
            <div className="timeline-header">
              <strong>{stage.stage}</strong>
              <span>{formatMilliseconds(stage.duration_ms)}</span>
            </div>
            <p>{stage.detail}</p>
            <small>{Object.entries(stage.meta).map(([key, value]) => `${humanize(key)}: ${String(value)}`).join(" · ")}</small>
          </div>
        </div>
      ))}
    </div>
  );
}

function HotspotCards({ hotspots }: { hotspots: Hotspot[] }) {
  if (hotspots.length === 0) {
    return <PanelPlaceholder label="No hotspots available for the current bridge" />;
  }
  return (
    <div className="hotspot-grid">
      {hotspots.map((hotspot) => (
        <div key={hotspot.zone} className="hotspot-card">
          <span className="hotspot-dot" style={{ background: riskColor(hotspot.max_probability ?? 0) }} />
          <div className="hotspot-copy">
            <div className="hotspot-head">
              <strong>{hotspot.zone}</strong>
              <span>{hotspot.hit_count} hits</span>
            </div>
            <small>Peak probability {formatPercent(hotspot.max_probability ?? 0)}</small>
            <small>Mean stress {formatDecimal(hotspot.mean_stress ?? 0)}</small>
          </div>
        </div>
      ))}
    </div>
  );
}

function XaiList({ factors }: { factors: XaiFactor[] }) {
  if (factors.length === 0) {
    return <PanelPlaceholder label="Explainability data is not available" />;
  }
  const maxImpact = Math.max(...factors.map((factor) => Math.abs(factor.impact ?? 0)), 0.000001);
  const totalImpact = factors.reduce((sum, factor) => sum + Math.abs(factor.impact ?? 0), 0) || 1;
  return (
    <div className="xai-list">
      {factors.map((factor) => (
        <div key={factor.feature} className="xai-item">
          <div className="xai-feature-block">
            <strong>{humanizeFeatureName(factor.feature)}</strong>
          </div>
          <div className="xai-context-block">
            <small>
              Baseline {formatPercent(factor.baseline_probability ?? 0)} → counterfactual {formatPercent(factor.counterfactual_probability ?? 0)}
            </small>
            <small className="xai-contribution-note">
              Contribution share {((Math.abs(factor.impact ?? 0) / totalImpact) * 100).toFixed(1)}%
            </small>
          </div>
          <div className="xai-bar">
            <div style={{ width: `${(Math.abs(factor.impact ?? 0) / maxImpact) * 100}%` }} />
          </div>
          <em className="xai-impact-value">{formatImpactValue(factor.impact)}</em>
        </div>
      ))}
    </div>
  );
}

function ValidationPanel({
  validation
}: {
  validation: BridgeDetail["validation"];
}) {
  return (
    <div className="validation-shell">
      <div className="validation-kpis">
        <InfoCard label="Verification Status" value={humanize(validation.status)} detail="Overall deployment-readiness classification for this output." compact />
        <InfoCard label="Evidence Score" value={`${validation.score}/100`} detail="Composite score across confidence, agreement, explainability, and consistency." compact />
        <InfoCard label="Confidence Band" value={validation.confidence_band} detail="Strength of the model's anomaly confidence profile." compact />
        <InfoCard label="Consensus Level" value={validation.consensus_level} detail="How strongly multiple signals support the alert." compact />
      </div>

      <div className="validation-grid">
        <div className="panel">
          <PanelHeader title="Verification Summary" subtitle="How this workspace justifies that the model output is operationally credible." />
          <NarrativeList items={validation.evidence_summary} />
        </div>
        <div className="panel">
          <PanelHeader title="Evidence Checks" subtitle="Each gate must pass or be reviewed before actioning the model output in production." />
          <div className="validation-check-stack">
            {validation.checks.map((check) => (
              <div key={check.name} className="validation-check-card">
                <div className="validation-check-head">
                  <strong>{check.name}</strong>
                  <span className={`risk-pill ${validationPillClass(check.status)}`}>{humanize(check.status)}</span>
                </div>
                <div className="category-bar-track">
                  <div className="category-bar-fill" style={{ width: `${Math.max(6, check.score)}%` }} />
                </div>
                <small>{check.detail}</small>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function AnomalyTable({ rows }: { rows: AnomalyRow[] }) {
  const [expanded, setExpanded] = useState(false);
  if (rows.length === 0) {
    return <PanelPlaceholder label="No anomaly rows available" />;
  }
  const previewCount = 10;
  const visibleRows = expanded ? rows : rows.slice(0, previewCount);

  return (
    <div className="table-preview-shell">
      <div className="table-shell">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Probability</th>
              <th>Hotspot</th>
              <th>Deflection</th>
              <th>Displacement</th>
              <th>Vibration</th>
              <th>PoF</th>
              <th>SHI</th>
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row) => (
              <tr key={`${row.timestamp}-${row.hotspot}`}>
                <td>{row.timestamp ? formatDate(row.timestamp) : "-"}</td>
                <td>{formatPercent(row.anomaly_probability ?? 0)}</td>
                <td>{row.hotspot}</td>
                <td>{formatDecimal(row.deflection_mm)}</td>
                <td>{formatDecimal(row.displacement_mm)}</td>
                <td>{formatDecimal(row.vibration_ms2)}</td>
                <td>{formatDecimal(row.failure_probability)}</td>
                <td>{formatDecimal(row.health_index)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > previewCount && (
        <button className="expand-button" onClick={() => setExpanded((value) => !value)}>
          {expanded ? "Show Fewer Rows" : `Show All ${rows.length} Rows`}
        </button>
      )}
    </div>
  );
}

function ReportsList({ reports }: { reports: OverviewResponse["reports"] }) {
  return (
    <div className="report-library">
      {reports.map((entry) => (
        <div key={entry.name}>{entry.title}</div>
      ))}
    </div>
  );
}

function InfoCard({
  label,
  value,
  detail,
  compact = false
}: {
  label: string;
  value: string;
  detail: string;
  compact?: boolean;
}) {
  return (
    <div className={`info-card ${compact ? "compact" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function PanelHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="panel-header">
      <div>
        <h3>{title}</h3>
        <p>{subtitle}</p>
      </div>
    </div>
  );
}

function StatList({ items }: { items: Array<[string, string]> }) {
  return (
    <div className="stat-list">
      {items.map(([label, value]) => (
        <div key={label} className="stat-row">
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function ImageCard({ title, src, wide = false }: { title: string; src: string | null; wide?: boolean }) {
  return (
    <div className={`image-card ${wide ? "wide" : ""}`}>
      <span>{title}</span>
      {src ? <img src={src} alt={title} /> : <div className="image-placeholder">Unavailable</div>}
    </div>
  );
}

function MetricChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-chip">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PanelPlaceholder({ label, tall = false }: { label: string; tall?: boolean }) {
  return <div className={`placeholder ${tall ? "tall" : ""}`}>{label}</div>;
}

function sectionTitle(section: WorkspaceSection) {
  switch (section) {
    case "bridge":
      return "Bridge Operations Panel";
    case "telemetry":
      return "Telemetry Analysis Panel";
    case "insar":
      return "InSAR Exploration Panel";
    case "reports":
      return "Engineering Reports Panel";
    default:
      return "Executive Monitoring Dashboard";
  }
}

function sectionCopy(section: WorkspaceSection, bridgeName: string) {
  switch (section) {
    case "bridge":
      return `Inspect ${bridgeName} in a 3D operations view with hotspot markers and runtime traces.`;
    case "telemetry":
      return `Open focused telemetry views for ${bridgeName} across GNSS, InSAR, and sensor channels.`;
    case "insar":
      return `Step through deformation frames and investigate imagery products for ${bridgeName}.`;
    case "reports":
      return "Review engineering reports and export-ready written summaries for stakeholders.";
    default:
      return "Review system posture, bridge priorities, and anomaly performance from an administrative perspective.";
  }
}

function riskColor(value: number) {
  if (value >= 0.95) {
    return "#c75a36";
  }
  if (value >= 0.7) {
    return "#d58b3f";
  }
  return "#4f8763";
}

function formatPercent(value: number | null) {
  if (value === null || Number.isNaN(value)) {
    return "-";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function formatDecimal(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(new Date(value));
}

function formatMilliseconds(value: number | null) {
  if (value === null || Number.isNaN(value)) {
    return "-";
  }
  return `${value.toFixed(2)} ms`;
}

function formatImpactValue(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  const absolute = Math.abs(value);
  if (absolute > 0 && absolute < 0.0001) {
    return value.toExponential(2);
  }
  return value.toFixed(6);
}

function asMetric(value: unknown) {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

function humanize(value: string) {
  return value.replace(/_/g, " ");
}

function validationPillClass(value: string) {
  if (value === "pass") return "low";
  if (value === "watch") return "medium";
  return "high";
}

function humanizeFeatureName(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\bP o F\b/g, "PoF")
    .replace(/\bS H I\b/g, "SHI");
}

function getErrorMessage(error: unknown) {
  if (error instanceof Error) {
    return error.message;
  }
  return "Unexpected application error";
}

export default App;
