export type BridgeCard = {
  bridge_id: string;
  bridge_name: string;
  city: string;
  region: string;
  lat: number;
  lon: number;
  anomaly_count: number;
  avg_probability: number;
  max_probability: number;
  mean_health_index: number;
  peak_deflection_mm: number;
  peak_displacement_mm: number;
  top_hotspot: string;
};

export type OverviewResponse = {
  fleet_metrics: {
    total_bridges: number;
    bridges_with_alerts: number;
    peak_probability: number;
    average_probability: number;
    highest_risk_bridge: string;
  };
  bridges: BridgeCard[];
  model_metrics: Record<string, number | string | null | Array<Record<string, unknown>>>;
  reports: ReportSummary[];
};

export type ReportSummary = {
  name: string;
  title: string;
  updated_at: string;
  size_bytes: number;
  kind?: string;
  provider?: string | null;
  model?: string | null;
};

export type ReportKpi = {
  label: string;
  value: string;
  detail: string;
};

export type PriorityAction = {
  bridge_id: string;
  bridge_name: string;
  risk_level: string;
  urgency_score: number;
  recommendation: string;
  rationale: string;
  dominant_hotspot: string;
  primary_driver: string;
  modalities_agreeing: string[];
  anomaly_count: number;
  max_probability: number | null;
};

export type AgentPanel = {
  agent: string;
  headline: string;
  body: string;
  bullets: string[];
};

export type CrewDashboard = {
  hero: {
    title: string;
    message: string;
    summary_points: string[];
    watchlist: string[];
  };
  kpis: ReportKpi[];
  priority_actions: PriorityAction[];
  risk_distribution: Array<{ label: string; count: number }>;
  recommendation_distribution: Array<{ label: string; count: number }>;
  driver_breakdown: Array<{ bridge_name: string; driver: string; urgency_score: number; risk_level: string }>;
  bridge_risk_series: Array<{ bridge_name: string; max_probability: number | null; anomaly_count: number; top_hotspot: string }>;
  hotspot_mix: Array<{ label: string; count: number }>;
  agent_panels: AgentPanel[];
  chart_annotations: string[];
  operator_prompts: string[];
  maintenance_queue: string[];
  markdown: string;
};

export type ReportDetail = {
  name: string;
  content: string;
  updated_at: string;
  kind?: string;
  metadata?: Record<string, unknown>;
  dashboard?: CrewDashboard | null;
};

export type TimePoint = Record<string, number | string | null>;

export type InSarFrame = {
  timestamp: string | null;
  image_path: string | null;
  mask_path: string | null;
  overlay_path: string | null;
  interferogram_path: string | null;
  heatmap_path: string | null;
  coherence_path: string | null;
  mask_ratio: number | null;
  deformation_energy: number | null;
  coherence_mean: number | null;
};

export type XaiFactor = {
  feature: string;
  impact: number | null;
  baseline_probability: number | null;
  counterfactual_probability: number | null;
};

export type ValidationCheck = {
  name: string;
  status: string;
  score: number;
  detail: string;
};

export type ValidationSummary = {
  status: string;
  score: number;
  confidence_band: string;
  consensus_level: string;
  evidence_summary: string[];
  checks: ValidationCheck[];
};

export type Hotspot = {
  zone: string;
  x: number;
  y: number;
  hit_count: number;
  max_probability: number | null;
  mean_stress: number | null;
};

export type RuntimeStage = {
  stage: string;
  detail: string;
  duration_ms: number | null;
  meta: Record<string, unknown>;
};

export type AnomalyRow = {
  timestamp: string | null;
  anomaly_probability: number | null;
  anomaly: number;
  hotspot: string;
  strain_hotspot: number | null;
  deflection_mm: number | null;
  displacement_mm: number | null;
  vibration_ms2: number | null;
  failure_probability: number | null;
  health_index: number | null;
};

export type BridgeDetail = {
  bridge: {
    bridge_id: string;
    bridge_name: string;
    city: string;
    region: string;
    lat: number;
    lon: number;
    latest_timestamp: string | null;
    anomaly_count: number;
    max_probability: number;
    avg_probability: number;
    mean_health_index: number;
    peak_deflection_mm: number;
    peak_displacement_mm: number;
    top_hotspot: string;
  };
  telemetry: {
    gnss: TimePoint[];
    insar: TimePoint[];
    sensors: TimePoint[];
  };
  insar_frames: InSarFrame[];
  xai_factors: XaiFactor[];
  validation: ValidationSummary;
  hotspots: Hotspot[];
  anomalies: AnomalyRow[];
  runtime_trace: RuntimeStage[];
  report_count: number;
};

export type OperationStep = {
  stage: string;
  detail: string;
  status: string;
  meta: Record<string, unknown>;
  timestamp: string;
};

export type OperationStatus = {
  id: string;
  kind: string;
  target: string | null;
  status: string;
  message: string;
  progress: number;
  steps: OperationStep[];
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  updated_at: string;
};
