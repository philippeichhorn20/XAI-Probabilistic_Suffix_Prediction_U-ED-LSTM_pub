"""Pydantic request/response models for the Interactive Case Explorer API."""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ============================================================
# Feature metadata
# ============================================================

class CategoricalFeatureInfo(BaseModel):
    name: str
    num_values: int
    idx_to_value: Dict[int, str]
    value_to_idx: Dict[str, int]


class NumericalFeatureInfo(BaseModel):
    name: str


# ============================================================
# Event / Case schemas
# ============================================================

class EventSchema(BaseModel):
    categorical: Dict[str, int]
    numerical: Dict[str, float]


class EditableCaseSchema(BaseModel):
    events: List[EventSchema]
    cat_feature_names: List[str]
    num_feature_names: List[str]
    original_length: int


# ============================================================
# Response: GET /api/models
# ============================================================

class ModelInfo(BaseModel):
    name: str
    display_name: str


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


# ============================================================
# Response: GET /api/models/{name}/cases
# ============================================================

class CasesResponse(BaseModel):
    cases: List[str]
    total: int
    offset: int
    limit: int


# ============================================================
# Response: GET /api/models/{name}/cases/{case}
# ============================================================

class CaseLoadResponse(BaseModel):
    case: EditableCaseSchema
    cat_info: List[CategoricalFeatureInfo]
    num_info: List[NumericalFeatureInfo]
    concept_name: str
    case_level_cat: List[str]
    case_level_num: List[str]


# ============================================================
# Request/Response: POST /api/models/{name}/predict
# ============================================================

class PredictRequest(BaseModel):
    case: EditableCaseSchema
    prefix_length: int
    max_suffix_length: int = 15
    top_k: int = 5


class TopKPrediction(BaseModel):
    activity: str
    probability: float


class PredictionStepResult(BaseModel):
    predicted_activity: str
    top_k: List[TopKPrediction]


class PredictResponse(BaseModel):
    predicted_activities: List[str]
    predicted_sequence: str
    steps: List[PredictionStepResult]
    actual_suffix_activities: Optional[List[str]] = None


# ============================================================
# Request/Response: POST /api/models/{name}/attribution
# ============================================================

class AttributionRequest(BaseModel):
    case: EditableCaseSchema
    prefix_length: int
    target: Optional[str] = None  # defaults to concept_name
    target_class: str = "auto"
    suffix_step: int = 0
    n_steps: int = 50
    method: str = "integrated_gradients"


class AttributionResponse(BaseModel):
    attributions: Dict[str, List[float]]  # feature_name -> attribution per step
    feature_names: List[str]
    step_labels: List[str]
    target_description: str
    convergence_delta: Optional[float] = None
    prefix_length: int
