"""Configuration for available models in the Interactive Case Explorer.

The AVAILABLE_MODELS list at the bottom is built dynamically from the per-dataset
configs in ``src/interpretability/config/`` so paths live in one place. Each entry
declares its ``model_class`` so the manager + predictor can dispatch correctly:

- ``henryk_uedlstm``: U-ED-LSTM (DropoutUncertaintyEncoderDecoderLSTM). Default.
- ``camargo_join``: FullShared_Join_LSTM.
- ``camargo_sharedcat``: SharedCat_LSTM.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
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

    # Loader / predictor dispatch.
    model_class: str = "henryk_uedlstm"
    # Camargo-only: which dataset cat/num positions the model consumes, and the
    # n-gram window size (>0 means right-align the last N events at predict time).
    cat_indices: Optional[Tuple[int, ...]] = None
    num_indices: Optional[Tuple[int, ...]] = None
    ngram_size: int = 0

    def get_model_path(self, project_root: Path) -> Path:
        return project_root / self.model_path

    def get_test_data_path(self, project_root: Path) -> Path:
        return project_root / self.test_data_path

    def get_csv_path(self, project_root: Path) -> Path:
        if self.csv_path is None:
            return None
        return project_root / self.csv_path


# ============================================================
# AVAILABLE MODELS
#
# Built dynamically from src/interpretability/config/<dataset>_config.py so
# paths live in one place. Each per-dataset config exposes both henryk
# (U-ED-LSTM) old/improved checkpoints and the Camargo baseline. We register
# whichever variants have actual paths configured.
# ============================================================

# Per-dataset metadata that's not in the per-dataset configs (CSV paths,
# case-level feature lists, csv_case_col). Keyed by dataset_name lowercase.
_DATASET_EXTRAS = {
    "helpdesk": dict(
        case_level_cat=["VariantIndex", "customer", "product", "responsible_section",
                        "seriousness", "seriousness_2", "service_level", "service_type",
                        "support_section", "workgroup"],
        csv_path="data/helpdesk.csv",
        csv_case_col="Case ID",
    ),
    "sepsis": dict(
        case_level_cat=None,
        csv_path="data/Sepsis.csv",
        csv_case_col="case:concept:name",
    ),
    "domestic_declarations": dict(
        case_level_cat=None,
        csv_path="data/domestic_declarations.csv",
        csv_case_col="Case ID",
    ),
    "bpic17": dict(
        case_level_cat=["case:LoanGoal", "case:ApplicationType"],
        case_level_num=["case:RequestedAmount"],
        csv_path="data/BPI_Challenge_2017.csv",
        csv_case_col="case:concept:name",
    ),
}


def _build_available_models() -> List[ModelConfig]:
    """Construct AVAILABLE_MODELS by reading per-dataset configs."""
    from src.interpretability.config import (
        sepsis_config, helpdesk_config, bpic17_config, domestic_declarations_config,
    )

    entries: List[ModelConfig] = []
    dataset_modules = [
        ("sepsis", sepsis_config.CONFIG),
        ("helpdesk", helpdesk_config.CONFIG),
        ("bpic17", bpic17_config.CONFIG),
        ("domestic_declarations", domestic_declarations_config.CONFIG),
    ]

    for ds_key, c in dataset_modules:
        extras = _DATASET_EXTRAS.get(ds_key, {})
        common = dict(
            concept_name=c.concept_name,
            all_cat=list(c.all_cat),
            all_num=list(c.all_num),
            growing_num_values=list(c.growing_num_values),
            case_level_cat=extras.get("case_level_cat"),
            case_level_num=extras.get("case_level_num"),
            csv_path=extras.get("csv_path"),
            csv_case_col=extras.get("csv_case_col", "Case ID"),
        )

        # --- Henryk U-ED-LSTM, old variant ---
        if c.model_path_old and c.test_data_path_old:
            entries.append(ModelConfig(
                name=f"{ds_key}_henryk_old",
                display_name=f"{c.dataset_name} — Henryk (old)",
                model_path=c.model_path_old,
                test_data_path=c.test_data_path_old,
                model_class="henryk_uedlstm",
                **common,
            ))

        # --- Henryk U-ED-LSTM, improved variant (when checkpoint exists) ---
        if c.model_path_improved and c.model_path_improved != "None" \
                and c.test_data_path_improved:
            entries.append(ModelConfig(
                name=f"{ds_key}_henryk_improved",
                display_name=f"{c.dataset_name} — Henryk (improved)",
                model_path=c.model_path_improved,
                test_data_path=c.test_data_path_improved,
                model_class="henryk_uedlstm",
                **common,
            ))

        # --- Camargo baseline (one variant per dataset) ---
        if c.camargo_model_pickle and c.camargo_test_pickle:
            cls_map = {
                "FullShared_Join_LSTM": "camargo_join",
                "SharedCat_LSTM": "camargo_sharedcat",
            }
            entries.append(ModelConfig(
                name=f"{ds_key}_camargo",
                display_name=f"{c.dataset_name} — Camargo",
                model_path=c.camargo_model_pickle,
                test_data_path=c.camargo_test_pickle,
                model_class=cls_map.get(c.camargo_model_class, "camargo_join"),
                cat_indices=tuple(c.camargo_cat_indices) if c.camargo_cat_indices else None,
                num_indices=tuple(c.camargo_num_indices) if c.camargo_num_indices else None,
                ngram_size=int(c.camargo_ngram_size or 0),
                **common,
            ))

    return entries


AVAILABLE_MODELS: List[ModelConfig] = _build_available_models()


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
