"""
Configuration for LORELEY algorithm.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LoreleyConfig:
    """
    Configuration for LORELEY algorithm.

    Attributes:
        population_size: Number of individuals per class in genetic algorithm
        n_generations: Number of generations to evolve
        crossover_prob: Probability of crossover between parents
        mutation_prob: Probability of mutation for each offspring
        edit_distance_threshold: Max Levenshtein distance for similar prefixes
        max_tree_depth: Maximum depth of decision tree (None = unlimited)
        min_samples_leaf: Minimum samples required at each leaf
        test_ratio: Fraction of synthetic data for fidelity evaluation
        seed: Random seed for reproducibility
    """

    # Genetic algorithm parameters
    population_size: int = 600  # Per class
    n_generations: int = 15
    crossover_prob: float = 0.2
    mutation_prob: float = 0.7

    # Similarity threshold for edit distance (0 = same length only)
    edit_distance_threshold: int = 5

    # Decision tree parameters
    max_tree_depth: Optional[int] = 5
    min_samples_leaf: int = 5

    # Train/test split for fidelity evaluation
    test_ratio: float = 0.2

    # Random seed
    seed: int = 42
