"""
Interpretability module for the suffix prediction model.

This module provides tools for explaining model predictions using:
- Integrated Gradients (IG)
- SHAP (Shapley Additive Explanations)
- ICE (Individual Conditional Expectations)

Usage:
    from interpretability import InterpretabilityTool

    tool = InterpretabilityTool(model, data_set_categories)

    # Compute attribution map
    attr_map = tool.compute_attribution_map(
        process=process,
        prefix_length=5,
        target='Activity',
        target_class='auto',
        method='integrated_gradients'  # or 'shap' or 'ice'
    )

    # Output formats
    print(attr_map.to_string())   # Text
    fig = attr_map.plot()          # Matplotlib figure
    tensor = attr_map.to_tensor()  # Raw tensor

Interactive Explorer:
    python -m interpretability.attribution_explorer
    # Opens a web interface at http://localhost:7860

Notebook Workflow:
    1. Run notebooks/helpdesk/data_and_metrics.ipynb - Load data, compute metrics
    2. Run notebooks/helpdesk/single_attribution_analysis.ipynb - Individual case analysis
    3. Run notebooks/helpdesk/aggregated_attribution_analysis.ipynb - Population analysis
    4. (Optional) Run notebooks/helpdesk/method_comparison.ipynb - Compare IG/SHAP/LIME

Configuration:
    To switch datasets, edit the config import in each notebook:
        from src.interpretability.config.helpdesk_config import CONFIG
        # or
        from src.interpretability.config.bpic2012_config import CONFIG
"""

from .attribution.integrated_gradients import IntegratedGradients
from .model.target_selectors import (
    TargetSelector,
    CategoricalTargetSelector,
    NumericalTargetSelector,
    create_target_selector
)
from .model.baselines import BaselineGenerator, ZeroBaseline, PaddingBaseline, MeanBaseline
from .model.model_wrapper import IGModelWrapper
from .visualization.attribution_vis import AttributionVisualizer
from .interpretability_tool import InterpretabilityTool, AttributionMap
from .attribution.shap_explainer import SequenceSHAP
from .attribution.ice_explainer import SequenceICE
from .utils.sequence_utils import (
    SequenceFormatter,
    SequenceChange,
    format_sequence,
    format_comparison,
    print_sequence,
    print_comparison
)
from .utils.tensor_decoder import TensorDecoder

__all__ = [
    'InterpretabilityTool',
    'AttributionMap',
    'SequenceSHAP',
    'SequenceICE',
    'IntegratedGradients',
    'TargetSelector',
    'CategoricalTargetSelector',
    'NumericalTargetSelector',
    'create_target_selector',
    'BaselineGenerator',
    'ZeroBaseline',
    'PaddingBaseline',
    'MeanBaseline',
    'IGModelWrapper',
    'AttributionVisualizer',
    # Sequence utilities
    'SequenceFormatter',
    'SequenceChange',
    'format_sequence',
    'format_comparison',
    'print_sequence',
    'print_comparison',
    # Tensor decoding
    'TensorDecoder',
]
