"""
Genetic algorithm components for LORELEY.

Implements control-flow-aware crossover and mutation operations that
treat activity frequency features as a unit to maintain process structure.
"""

import numpy as np
import random
from typing import List, Tuple, Dict


class GeneticOperators:
    """
    Genetic operators for neighborhood generation.

    Control flow features (activity frequencies) are treated as a single unit
    during crossover and mutation to preserve process structure.
    """

    @staticmethod
    def crossover(parent1: np.ndarray,
                  parent2: np.ndarray,
                  control_flow_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Control-flow-aware two-point crossover.

        Control flow features are swapped as a single unit with 50% probability.
        Other features undergo standard two-point crossover.

        Args:
            parent1: First parent feature vector
            parent2: Second parent feature vector
            control_flow_indices: Indices of control flow features

        Returns:
            Two child feature vectors
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Swap control flow as a unit with 50% probability
        if random.random() < 0.5:
            child1[control_flow_indices] = parent2[control_flow_indices]
            child2[control_flow_indices] = parent1[control_flow_indices]

        # Standard two-point crossover for other features
        other_indices = [i for i in range(len(parent1)) if i not in control_flow_indices]
        if len(other_indices) >= 2:
            pt1, pt2 = sorted(random.sample(range(len(other_indices)), 2))
            swap_indices = [other_indices[i] for i in range(pt1, pt2)]
            child1[swap_indices] = parent2[swap_indices]
            child2[swap_indices] = parent1[swap_indices]

        return child1, child2

    @staticmethod
    def mutate(individual: np.ndarray,
               empirical_dist: Dict[int, np.ndarray],
               control_flow_indices: List[int],
               mutation_rate: float = 0.3) -> np.ndarray:
        """
        Control-flow-aware mutation.

        Control flow features are mutated together from the empirical distribution.
        Other features are mutated independently.

        Args:
            individual: Feature vector to mutate
            empirical_dist: Empirical distribution of feature values
            control_flow_indices: Indices of control flow features
            mutation_rate: Per-feature mutation probability for non-control-flow features

        Returns:
            Mutated feature vector
        """
        mutated = individual.copy()

        # Mutate control flow as a unit (swap with another individual's control flow)
        if random.random() < 0.5 and len(empirical_dist) > 0:
            # Pick a random control flow from empirical distribution
            n_samples = len(empirical_dist.get(0, []))
            if n_samples > 0:
                rand_idx = random.randint(0, n_samples - 1)
                for cf_idx in control_flow_indices:
                    if cf_idx in empirical_dist and len(empirical_dist[cf_idx]) > rand_idx:
                        mutated[cf_idx] = empirical_dist[cf_idx][rand_idx]

        # Mutate other features independently
        other_indices = [i for i in range(len(individual)) if i not in control_flow_indices]
        for idx in other_indices:
            if random.random() < mutation_rate:
                if idx in empirical_dist and len(empirical_dist[idx]) > 0:
                    mutated[idx] = random.choice(empirical_dist[idx])

        return mutated

    @staticmethod
    def fitness(z: np.ndarray,
                x: np.ndarray,
                target_class: int,
                predicted_class: int = None) -> float:
        """
        Fitness function for genetic algorithm.

        Maximizes similarity to x while preferring instances with target class.

        fitness = 1_{b(z)=target} + (1 - d(x,z)) - 1_{x=z}

        Args:
            z: Candidate feature vector
            x: Original instance feature vector
            target_class: Target class for counterfactual
            predicted_class: Predicted class of z (if known)

        Returns:
            Fitness score (higher is better)
        """
        # Normalized distance (0 = identical, 1 = very different)
        d = np.linalg.norm(x - z) / (np.linalg.norm(x) + 1e-8)
        d = min(d, 1.0)  # Clip to [0, 1]

        # Check if identical (penalize exact copies)
        is_identical = np.allclose(x, z, rtol=1e-5)

        # Class match bonus (if we know the predicted class)
        if predicted_class is not None:
            class_match = 1.0 if predicted_class == target_class else 0.0
        else:
            class_match = 1.0  # Assume target class for synthetic instances

        return class_match + (1 - d) - (1.0 if is_identical else 0.0)


class NeighborhoodGenerator:
    """
    Generates synthetic neighborhood using genetic algorithm.
    """

    def __init__(self,
                 population_size: int = 600,
                 n_generations: int = 15,
                 crossover_prob: float = 0.2,
                 mutation_prob: float = 0.7,
                 seed: int = 42):
        """
        Args:
            population_size: Number of individuals per class
            n_generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            seed: Random seed
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        random.seed(seed)
        np.random.seed(seed)

    def compute_empirical_distribution(self, population: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute empirical distribution of feature values for mutation.

        Args:
            population: Population of feature vectors [n_samples, n_features]

        Returns:
            Dictionary mapping feature index to array of observed values
        """
        distributions = {}
        for i in range(population.shape[1]):
            distributions[i] = population[:, i].copy()
        return distributions

    def evolve(self,
               x: np.ndarray,
               init_population: np.ndarray,
               target_class_idx: int,
               control_flow_indices: List[int]) -> np.ndarray:
        """
        Evolve population towards target class.

        Args:
            x: Original instance to explain
            init_population: Initial population from similar prefixes
            target_class_idx: Target class for this population
            control_flow_indices: Indices of control flow features

        Returns:
            Final evolved population
        """
        # Initialize population
        if len(init_population) >= self.population_size:
            indices = np.random.choice(len(init_population), self.population_size, replace=False)
            population = init_population[indices].copy()
        else:
            # Pad with copies and noise
            n_copies = (self.population_size // len(init_population)) + 1
            population = np.tile(init_population, (n_copies, 1))[:self.population_size]
            # Add noise
            noise = np.random.normal(0, 0.1, population.shape)
            population = population + noise

        # Compute empirical distribution for mutation
        empirical_dist = self.compute_empirical_distribution(init_population)

        for gen in range(self.n_generations):
            # Calculate fitness for each individual
            fitness = np.array([
                GeneticOperators.fitness(ind, x, target_class_idx)
                for ind in population
            ])

            # Select top individuals (tournament selection)
            n_select = self.population_size // 2
            selected_indices = np.argsort(fitness)[-n_select:]
            selected = population[selected_indices]

            # Crossover
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                if random.random() < self.crossover_prob:
                    child1, child2 = GeneticOperators.crossover(
                        selected[i], selected[i + 1], control_flow_indices
                    )
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([selected[i].copy(), selected[i + 1].copy()])

            # Mutation
            for i in range(len(offspring)):
                if random.random() < self.mutation_prob:
                    offspring[i] = GeneticOperators.mutate(
                        offspring[i], empirical_dist, control_flow_indices
                    )

            # Form new population
            offspring = np.array(offspring)
            population = np.vstack([selected, offspring])[:self.population_size]

        return population
