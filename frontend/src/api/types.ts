// Mirrors backend/app/api/schemas.py. Keep the two in sync.

export type Label = "parkinsons_like" | "healthy_like";
export type Group = "parkinsons" | "healthy" | "neither";
export type Percentiles = Record<"p5" | "p25" | "p50" | "p75" | "p95", number>;

export interface PredictionOut {
  label: Label;
  label_text: string;
  probability: number;
  base_rate: number;
  reference_probability: number;
}

export interface Contribution {
  feature: string;
  value: number;
  contribution: number;
}

export interface FeatureRange {
  feature: string;
  value: number;
  parkinsons: Percentiles;
  healthy: Percentiles;
  within_training_range: boolean;
  closer_to: Group;
}

export interface PredictionResponse {
  prediction: PredictionOut;
  contributions: Contribution[];
  feature_ranges: FeatureRange[];
  features: Record<string, number>;
  warnings: string[];
  model: { type: string; version: string };
  disclaimer: string;
  privacy: string;
}

export interface ErrorBody {
  error: { code: string; message: string };
  disclaimer: string;
}
