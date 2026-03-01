"""
LORELEY: LOcal Rule-based Explanations for predictive process monitoring.

Generates counterfactual explanations using:
- Levenshtein edit distance for similar prefix finding
- Genetic algorithm for synthetic neighborhood generation
- Decision tree for rule extraction

Based on Huang et al. (2022).
"""

from .loreley import (
    LORELEY,
    LoreleyModelPredictor,
    create_loreley_for_model
)
from .config import LoreleyConfig
from .encoder import PrefixEncoder
from .explanation import LoreleyExplanation
from .genetic import GeneticOperators, NeighborhoodGenerator

__all__ = [
    # Main classes
    'LORELEY',
    'LoreleyConfig',
    'LoreleyExplanation',
    'LoreleyModelPredictor',
    'create_loreley_for_model',
    # Supporting classes
    'PrefixEncoder',
    'GeneticOperators',
    'NeighborhoodGenerator',
]
