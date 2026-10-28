export interface Machine {
  machine_id: string;
  machine_name: string;
  machine_type: string;
  installation_date?: string;
  location?: string;
  status: string;
  operating_hours: number;
  last_maintenance?: string;
  created_at?: string;
  last_prediction?: string;
}

export interface OverviewStats {
  total_machines: number;
  healthy: number;
  warning: number;
  critical: number;
  total_predictions: number;
  anomalies_detected: number;
  avg_failure_probability: number;
}

export interface FeatureContribution {
  feature: string;
  feature_value: number;
  shap_value: number;
  direction: "positive" | "negative";
}

export interface SHAPExplanation {
  base_value: number;
  top_features: FeatureContribution[];
  all_contributions: FeatureContribution[];
}

export interface PredictionResponse {
  machine_id?: string;
  failure_probability: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  prediction: "NORMAL" | "FAILURE_RISK";
  recommended_action: string;
  timestamp: string;
  model_version: string;
}

export interface FullAnalysisResponse extends PredictionResponse {
  is_anomaly: boolean;
  anomaly_score: number;
  anomaly_label: "Normal" | "Anomaly Detected";
  shap_explanation: SHAPExplanation;
}

export interface AnalyticsFailures {
  risk_distribution: Record<string, number>;
  prediction_counts: Record<string, number>;
  daily_failures_7d: { date: string; count: number }[];
  total_analyzed: number;
}

export interface AnomalyStats {
  total_analyzed: number;
  anomalies_count: number;
  normal_count: number;
  anomaly_rate: number;
  avg_anomaly_score: number;
  top_anomalous_machines: { machine_id: string; anomaly_count: number }[];
}

export interface ModelInfo {
  primary_model: string;
  model_version: string;
  feature_count: number;
  feature_names: string[];
  metrics: Record<string, any>;
  models_available: string[];
  is_ready: boolean;
  risk_thresholds: Record<string, string>;
}
