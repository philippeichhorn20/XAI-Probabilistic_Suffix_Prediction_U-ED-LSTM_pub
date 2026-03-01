"""IG attribution adapter bridging InterpretabilityTool for the API.

Converts between EditableCase tensors and the format expected by
InterpretabilityTool.compute_attribution_map().
"""

import numpy as np
from typing import Dict, Optional

from .case_editor import CaseEditor, EditableCase
from .model_manager import ModelManager
from .config import get_model_config
from .api_schemas import AttributionResponse

# Module-level cache: model_name -> InterpretabilityTool instance
_tool_cache: Dict[str, "InterpretabilityTool"] = {}


def _get_interpretability_tool(model_name: str, manager: ModelManager):
    """Get or create a cached InterpretabilityTool for the given model."""
    if model_name not in _tool_cache:
        from src.interpretability.interpretability_tool import InterpretabilityTool

        model = manager.load_model(model_name)
        tool = InterpretabilityTool(
            model=model,
            data_set_categories=model.data_set_categories,
        )
        _tool_cache[model_name] = tool
    return _tool_cache[model_name]


def compute_ig_attribution(
    model_name: str,
    manager: ModelManager,
    case: EditableCase,
    prefix_length: int,
    target: Optional[str] = None,
    target_class: str = "auto",
    suffix_step: int = 0,
    n_steps: int = 50,
    method: str = "integrated_gradients",
) -> AttributionResponse:
    """Compute attribution using InterpretabilityTool and return API response.

    1. Converts EditableCase -> 1D tensors via CaseEditor.editable_to_tensors
    2. Calls InterpretabilityTool.compute_attribution_map
    3. Serializes AttributionMap -> AttributionResponse
    """
    config = get_model_config(model_name)
    tool = _get_interpretability_tool(model_name, manager)

    if target is None:
        target = config.concept_name

    cat_info = manager.get_categorical_info(model_name)
    num_info = manager.get_numerical_info(model_name)
    cat_names = [c.name for c in cat_info]
    num_names = [n.name for n in num_info]

    editor = CaseEditor(cat_names, num_names)
    process = editor.editable_to_tensors(case)

    attr_map = tool.compute_attribution_map(
        process=process,
        prefix_length=prefix_length,
        target=target,
        target_class=target_class,
        suffix_scope="step",
        suffix_step=suffix_step,
        method=method,
        n_steps=n_steps,
    )

    # Serialize AttributionMap to response
    attributions_dict: Dict[str, list] = {}
    for feat_name in attr_map.feature_names:
        if feat_name in attr_map.attributions:
            attr = attr_map.attributions[feat_name]
            arr = np.array(attr).flatten().tolist() if not isinstance(attr, list) else attr
            attributions_dict[feat_name] = [float(v) for v in arr]
        else:
            attributions_dict[feat_name] = [0.0] * len(attr_map.step_labels)

    return AttributionResponse(
        attributions=attributions_dict,
        feature_names=attr_map.feature_names,
        step_labels=attr_map.step_labels,
        target_description=attr_map.target_description,
        convergence_delta=attr_map.convergence_delta,
        prefix_length=attr_map.prefix_length,
    )
