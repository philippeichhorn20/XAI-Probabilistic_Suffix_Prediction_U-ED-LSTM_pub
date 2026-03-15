"""Configuration for available models in the Interactive Case Explorer."""

from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    display_name: str
    model_path: str
    test_data_path: str
    concept_name: str
    all_cat: List[str]
    all_num: List[str]
    growing_num_values: List[str]
    # Features that are constant for the entire case (edited once, not per event)
    # If None, will be auto-detected from data
    case_level_cat: List[str] = None
    case_level_num: List[str] = None
    # Path to the source CSV (for recovering unseen encoded values)
    csv_path: str = None
    # Column name in CSV that holds the case ID
    csv_case_col: str = "Case ID"

    def get_model_path(self, project_root: Path) -> Path:
        return project_root / self.model_path

    def get_test_data_path(self, project_root: Path) -> Path:
        return project_root / self.test_data_path

    def get_csv_path(self, project_root: Path) -> Path:
        if self.csv_path is None:
            return None
        return project_root / self.csv_path


# ============================================================
# CONFIGURABLE MODEL LIST
# Add or remove models here
# ============================================================

AVAILABLE_MODELS: List[ModelConfig] = [
    ModelConfig(
        name="helpdesk",
        display_name="Helpdesk",
        model_path="src/notebooks/training_variational_dropout/Helpdesk/Helpdesk_full_grad_norm_new_4layer.pkl",
        test_data_path="encoded_data/test_philipp/helpdesk_all_5_test.pkl",
        concept_name="Activity",
        all_cat=["Activity", "Resource"],
        all_num=["case_elapsed_time", "event_elapsed_time"],
        growing_num_values=["case_elapsed_time"],
        # Case-level features (constant for entire case)
        case_level_cat=["VariantIndex", "customer", "product", "responsible_section",
                        "seriousness", "seriousness_2", "service_level", "service_type",
                        "support_section", "workgroup"],
        case_level_num=None,  # Auto-detect numerical
        csv_path="data/helpdesk.csv",
        csv_case_col="Case ID",
    ),
    ModelConfig(
        name="domestic_declarations",
        display_name="DomesticDeclarations",
        model_path="src/notebooks/training_variational_dropout/DomesticDeclarations/DomesticDeclarations_full_grad_norm_4layer.pkl",
        test_data_path="encoded_data/test_philipp/domestic_declarations_all_5_test.pkl",
        concept_name="Activity",
        all_cat=["Activity", "Resource"],
        all_num=["case_elapsed_time", "event_elapsed_time"],
        growing_num_values=["case_elapsed_time"],
        case_level_cat=None,
        case_level_num=None,
        csv_path="data/domestic_declarations.csv",
        csv_case_col="Case ID",
    ),
    ModelConfig(
        name="bpic17",
        display_name="BPIC17",
        model_path="src/notebooks/training_variational_dropout/BPIC17/BPIC_2017_full_grad_norm_new_4layer.pkl",
        test_data_path="encoded_data/BPIC_2017_all_5_test.pkl",
        concept_name="concept:name",
        all_cat=["concept:name", "Action", "org:resource", "EventOrigin", "lifecycle:transition",
                 "case:LoanGoal", "case:ApplicationType", "Accepted", "Selected"],
        all_num=["case_elapsed_time", "event_elapsed_time", "day_in_week", "seconds_in_day",
                 "case:RequestedAmount", "FirstWithdrawalAmount", "NumberOfTerms", "MonthlyCost", "CreditScore"],
        growing_num_values=["case_elapsed_time"],
        # Case-level features (constant for entire case)
        case_level_cat=["case:LoanGoal", "case:ApplicationType"],
        case_level_num=["case:RequestedAmount"],
        csv_path="data/BPI_Challenge_2017.csv",
        csv_case_col="case:concept:name",
    ),
]


def get_model_config(model_name: str) -> ModelConfig:
    """Get model config by name."""
    for config in AVAILABLE_MODELS:
        if config.name == model_name:
            return config
    raise ValueError(f"Model '{model_name}' not found. Available: {[m.name for m in AVAILABLE_MODELS]}")


def get_available_model_names() -> List[str]:
    """Get list of available model names."""
    return [m.name for m in AVAILABLE_MODELS]


def get_available_model_display_names() -> Dict[str, str]:
    """Get mapping of model name to display name."""
    return {m.name: m.display_name for m in AVAILABLE_MODELS}
