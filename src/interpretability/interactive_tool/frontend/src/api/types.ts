// Mirrors the Pydantic schemas in api_schemas.py

export interface CategoricalFeatureInfo {
  name: string;
  num_values: number;
  idx_to_value: Record<number, string>;
  value_to_idx: Record<string, number>;
}

export interface NumericalFeatureInfo {
  name: string;
}

export interface EventData {
  categorical: Record<string, number>;
  numerical: Record<string, number>;
}

export interface EditableCase {
  events: EventData[];
  cat_feature_names: string[];
  num_feature_names: string[];
  original_length: number;
}

export interface ModelInfo {
  name: string;
  display_name: string;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

export interface CasesResponse {
  cases: string[];
  total: number;
  offset: number;
  limit: number;
}

export interface CaseLoadResponse {
  case: EditableCase;
  cat_info: CategoricalFeatureInfo[];
  num_info: NumericalFeatureInfo[];
  concept_name: string;
  case_level_cat: string[];
  case_level_num: string[];
}

export interface TopKPrediction {
  activity: string;
  probability: number;
}

export interface PredictionStepResult {
  predicted_activity: string;
  top_k: TopKPrediction[];
}

export interface PredictResponse {
  predicted_activities: string[];
  predicted_sequence: string;
  steps: PredictionStepResult[];
  actual_suffix_activities: string[] | null;
}

export interface PredictRequest {
  case: EditableCase;
  prefix_length: number;
  max_suffix_length?: number;
  top_k?: number;
}

export interface AttributionRequest {
  case: EditableCase;
  prefix_length: number;
  target?: string;
  target_class?: string;
  suffix_step?: number;
  n_steps?: number;
  method?: string;
}

export interface AttributionResponse {
  attributions: Record<string, number[]>;
  feature_names: string[];
  step_labels: string[];
  target_description: string;
  convergence_delta: number | null;
  prefix_length: number;
}
