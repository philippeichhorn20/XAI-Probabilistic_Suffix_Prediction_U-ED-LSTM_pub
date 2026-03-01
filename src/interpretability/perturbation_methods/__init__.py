"""
Perturbation-based explainability methods for process prediction.

This package provides counterfactual explanation methods for predictive process analytics:

- **LORELEY** (Huang et al., 2022): Genetic algorithm + decision tree approach
  Generates rule-based explanations using local neighborhood generation.

- **REVISED+** (Stevens et al., 2024): LSTM-VAE + Declare constraints approach
  Generates feasible and plausible counterfactuals via manifold learning.

Both methods are adapted for next activity prediction.

Example usage:
    >>> from src.interpretability.perturbation_methods import RevisedPlus, RevisedPlusConfig
    >>> # From EventLogDataset (recommended, uses full traces as per paper)
    >>> revised = RevisedPlus.from_dataset(model, train_dataset, config)
    >>> # Or manually
    >>> revised = RevisedPlus(model, vocab_size, activity_names, max_len, config)
    >>> revised.fit(full_traces)
    >>> explanation = revised.explain(prefix, target_activity=5)
"""

# =============================================================================
# Shared Components
# =============================================================================

# Base classes
from .base import (
    BaseExplanation,
    BaseCounterfactual,
    BaseModelPredictor,
    BaseCounterfactualMethod,
    ProcessModelPredictor,
    # Metrics
    compute_proximity_l1,
    compute_proximity_l2,
    compute_sparsity_l0,
    compute_longest_common_prefix,
)

# Declare constraints (shared between methods)
from .declare import (
    DeclareTemplate,
    DeclareConstraint,
    DeclareConstraintChecker,
    DeclareConstraintMiner,
)

# =============================================================================
# LORELEY
# =============================================================================

try:
    from .loreley import (
        LORELEY,
        LoreleyConfig,
        LoreleyExplanation,
        LoreleyModelPredictor,
        PrefixEncoder,
        GeneticOperators,
        NeighborhoodGenerator,
        create_loreley_for_model,
    )
    _LORELEY_AVAILABLE = True
except ImportError as e:
    _LORELEY_AVAILABLE = False
    _LORELEY_ERROR = str(e)

# =============================================================================
# REVISED+
# =============================================================================

from .revised_plus import csv_to_xes, SimpleSequenceVAE, train_vae
from .revised_plus import (
    RevisedPlusConfig,
    RevisedPlusCounterfactual,
    RevisedPlusExplanation,
    RevisedPlus,
    RevisedPlusModelPredictor,
    create_revised_plus_for_model,
)

# Optional imports that require external dependencies
try:
    from .revised_plus import discover_constraints, check_conformance, load_event_log
    _MP_DECLARE_AVAILABLE = True
except ImportError:
    _MP_DECLARE_AVAILABLE = False

try:
    from .revised_plus import discover_mpdeclare, MPDeclareConstraint
    _RUM_AVAILABLE = True
except ImportError:
    _RUM_AVAILABLE = False

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base classes
    'BaseExplanation',
    'BaseCounterfactual',
    'BaseModelPredictor',
    'BaseCounterfactualMethod',
    'ProcessModelPredictor',
    # Metrics
    'compute_proximity_l1',
    'compute_proximity_l2',
    'compute_sparsity_l0',
    'compute_longest_common_prefix',
    # Declare constraints
    'DeclareTemplate',
    'DeclareConstraint',
    'DeclareConstraintChecker',
    'DeclareConstraintMiner',
]

# REVISED+ (always available)
__all__.extend([
    'csv_to_xes',
    'SimpleSequenceVAE',
    'train_vae',
    'RevisedPlusConfig',
    'RevisedPlusCounterfactual',
    'RevisedPlusExplanation',
    'RevisedPlus',
    'RevisedPlusModelPredictor',
    'create_revised_plus_for_model',
])

# REVISED+ optional (require Declare4Py)
if _MP_DECLARE_AVAILABLE:
    __all__.extend([
        'discover_constraints',
        'check_conformance',
        'load_event_log',
    ])

# REVISED+ optional (require RuM)
if _RUM_AVAILABLE:
    __all__.extend([
        'discover_mpdeclare',
        'MPDeclareConstraint',
    ])

# Add LORELEY exports if available
if _LORELEY_AVAILABLE:
    __all__.extend([
        'LORELEY',
        'LoreleyConfig',
        'LoreleyExplanation',
        'LoreleyModelPredictor',
        'PrefixEncoder',
        'GeneticOperators',
        'NeighborhoodGenerator',
        'create_loreley_for_model',
    ])


def check_loreley_available() -> bool:
    """Check if LORELEY is available."""
    return _LORELEY_AVAILABLE


def get_loreley_error() -> str:
    """Get error message if LORELEY is not available."""
    if _LORELEY_AVAILABLE:
        return ""
    return _LORELEY_ERROR
