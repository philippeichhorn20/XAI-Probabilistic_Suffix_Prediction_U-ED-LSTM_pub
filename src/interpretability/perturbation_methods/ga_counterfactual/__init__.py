"""GA-based Counterfactual Generation (Guidotti et al. 2024 adapted).

Genetic algorithm with constraint-preserving operators for generating
counterfactual explanations adapted for next-activity prediction.
"""

from .ga_counterfactual import (
    GACounterfactualConfig,
    GACounterfactualResult,
    GACounterfactualExplanation,
    GACounterfactual,
    create_ga_counterfactual_for_model,
)

__all__ = [
    'GACounterfactualConfig',
    'GACounterfactualResult',
    'GACounterfactualExplanation',
    'GACounterfactual',
    'create_ga_counterfactual_for_model',
]
