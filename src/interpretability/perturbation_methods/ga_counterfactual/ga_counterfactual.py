"""
GA-based Counterfactual Generation (Guidotti et al. 2024 adapted).

Genetic algorithm with constraint-preserving crossover and mutation operators
for generating counterfactual explanations, adapted from outcome prediction
to next-activity prediction for the U-ED-LSTM model.

A configurable set of categorical features is mutable (e.g. Activity, Resource,
Variant index). All other features (remaining categoricals + all numericals)
are kept frozen from the original prefix.

Declare constraints apply to the activity channel only. Other mutable features
are mutated freely.

Key innovation (Guidotti et al.): precompute which Declare constraints the
original prefix satisfies, then ensure crossover/mutation never violate those
(on the activity channel).

Usage:
    >>> from src.interpretability.perturbation_methods.ga_counterfactual import (
    ...     GACounterfactual, GACounterfactualConfig, create_ga_counterfactual_for_model,
    ... )
    >>> config = GACounterfactualConfig(
    ...     population_size=200, n_generations=50,
    ...     mutable_cat_indices=[0, 1, 2],  # Activity, Resource, Variant index
    ... )
    >>> ga = create_ga_counterfactual_for_model(model, dataset, activity_names, config, constraints)
    >>> explanation = ga.explain(cat_tensors, num_tensors)
"""

import time
import random
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

from ..revised_plus.revised_plus import RevisedPlusModelPredictor
from ..declare import DeclareConstraint, DeclareConstraintChecker, DataCondition
from ..base import compute_sparsity_l0


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GACounterfactualConfig:
    """Configuration for GA-based counterfactual generation."""

    # GA parameters
    population_size: int = 200
    n_generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.3
    tournament_size: int = 5
    elite_size: int = 5

    # Mutable features: indices into cat_tensors that the GA may mutate.
    # Index 0 is always Activity. Other indices depend on the dataset.
    # E.g. for Helpdesk: [0, 1, 2] = Activity, Resource, Variant index.
    # None means activity-only (backward compatible).
    mutable_cat_indices: Optional[List[int]] = None

    # Case-level features: indices (into cat_tensors) that are constant across
    # all positions in a sequence (e.g. Variant index). When mutated, all
    # positions are set to the same value. Must be a subset of mutable_cat_indices.
    case_level_cat_indices: Optional[List[int]] = None

    # Fitness weights (single-objective, lower = better)
    w_validity: float = 5.0
    w_proximity: float = 1.0
    w_sparsity: float = 1.0
    w_plausibility: float = 1.0
    w_conformance: float = 2.0

    # Output
    top_k: int = 5
    diversity_threshold: float = 0.3

    # General
    device: str = "cpu"
    activity_feature: str = "Activity"
    verbose: bool = True


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class GACounterfactualResult:
    """A single GA counterfactual with quality metrics."""

    activity_sequence: List[int]
    cat_sequence: List[torch.Tensor]
    num_sequence: Optional[torch.Tensor]
    counterfactual_prediction: int
    counterfactual_prediction_name: str
    counterfactual_probability: float
    proximity: float
    sparsity: int
    plausibility: float
    conformance: float
    fitness: float
    generation: int


@dataclass
class GACounterfactualExplanation:
    """Full GA counterfactual explanation result."""

    original_activity_sequence: List[int]
    original_prediction: int
    original_prediction_name: str
    original_probability: float
    prefix_len: int
    counterfactuals: List[GACounterfactualResult]
    n_generations_run: int
    n_candidates_evaluated: int
    n_valid_counterfactuals: int
    search_time_seconds: float
    n_constraints: int
    mutable_cat_indices: List[int] = field(default_factory=lambda: [0])

    def __str__(self) -> str:
        lines = [
            "GA Counterfactual Explanation",
            "=" * 60,
            f"Original: {self.original_prediction_name} "
            f"(idx={self.original_prediction}, p={self.original_probability:.3f})",
            f"Prefix length: {self.prefix_len} events",
            f"Mutable features: cat indices {self.mutable_cat_indices}",
            f"Constraints: {self.n_constraints}",
            f"Search: {self.n_generations_run} generations, "
            f"{self.n_candidates_evaluated} evaluated, "
            f"{self.n_valid_counterfactuals} valid, "
            f"{self.search_time_seconds:.1f}s",
            "",
        ]
        if not self.counterfactuals:
            lines.append("No counterfactuals found.")
        else:
            lines.append(f"Top {len(self.counterfactuals)} counterfactuals:")
            for i, cf in enumerate(self.counterfactuals):
                lines.append(
                    f"  [{i+1}] -> {cf.counterfactual_prediction_name} "
                    f"(p={cf.counterfactual_probability:.3f}) | "
                    f"prox={cf.proximity:.3f} sparse={cf.sparsity} "
                    f"plaus={cf.plausibility:.3f} conf={cf.conformance:.2f} "
                    f"fit={cf.fitness:.4f} gen={cf.generation}"
                )
        return "\n".join(lines)


# =============================================================================
# Individual representation
# =============================================================================

# An individual is a list of lists: one List[int] per mutable feature channel,
# each of length prefix_len.  E.g. with mutable_cat_indices=[0,1,2]:
#   individual[0] = activity values  (prefix_len ints)
#   individual[1] = resource values  (prefix_len ints)
#   individual[2] = variant values   (prefix_len ints)
Individual = List[List[int]]


def _clone_individual(ind: Individual) -> Individual:
    return [list(ch) for ch in ind]


# =============================================================================
# Main Orchestrator
# =============================================================================

class GACounterfactual:
    """
    GA-based counterfactual generation with constraint-preserving operators.

    Adapts Guidotti et al. 2024 for next-activity prediction. Mutates a
    configurable set of categorical features; all others are frozen.
    Declare constraints are enforced on the activity channel only.
    """

    def __init__(
        self,
        predictor: RevisedPlusModelPredictor,
        activity_names: List[str],
        constraints: Set[DeclareConstraint],
        training_sequences: List[List[int]],
        cat_vocab_sizes: List[int],
        mutable_cat_indices: List[int],
        config: GACounterfactualConfig,
        data_conditions: Optional[Dict[DeclareConstraint, DataCondition]] = None,
    ):
        self.predictor = predictor
        self.activity_names = activity_names
        self.constraints = constraints
        self.prefix_safe_constraints = set(
            DeclareConstraintChecker.prefix_safe_constraints(constraints)
        )
        self.training_sequences = training_sequences
        self.config = config
        self.data_conditions = data_conditions or {}

        # Mutable feature info
        self.mutable_cat_indices = mutable_cat_indices
        self.cat_vocab_sizes = cat_vocab_sizes
        # Valid values per mutable channel (skip 0 = padding)
        self.valid_values = {
            idx: list(range(1, cat_vocab_sizes[idx]))
            for idx in mutable_cat_indices
        }
        self.n_activities = len(activity_names)
        # Position of activity channel (index 0) inside the individual
        self.activity_channel = mutable_cat_indices.index(0)
        # Case-level channels: positions within the individual that must be
        # uniform (same value at every position in the sequence)
        case_level = set(config.case_level_cat_indices or [])
        self.case_level_channels = {
            mutable_cat_indices.index(ci)
            for ci in case_level if ci in mutable_cat_indices
        }

    def explain(
        self,
        cat_tensors: List[torch.Tensor],
        num_tensors: List[torch.Tensor],
        target_class: Optional[int] = None,
    ) -> GACounterfactualExplanation:
        """
        Generate counterfactual explanations for a prefix.

        Args:
            cat_tensors: List of [1, seq_len] categorical tensors (batched).
            num_tensors: List of [1, seq_len] numerical tensors (batched).
            target_class: Target activity index (None = any different class).

        Returns:
            GACounterfactualExplanation with diverse top-k counterfactuals.
        """
        t_start = time.time()
        cfg = self.config

        # --- Extract original values for all channels ---
        act_tensor = cat_tensors[0].squeeze(0)  # [seq_len]
        seq_len = act_tensor.shape[0]
        non_pad_mask = act_tensor != 0
        prefix_len = int(non_pad_mask.sum().item())
        pad_len = seq_len - prefix_len

        # Original values per mutable channel (non-padded)
        original: Individual = []
        for ci in self.mutable_cat_indices:
            t = cat_tensors[ci].squeeze(0)
            original.append(t[pad_len:].tolist())

        original_acts = original[self.activity_channel]

        # Frozen tensors: all cat channels NOT in mutable set + all num
        frozen_cat_indices = [
            i for i in range(len(cat_tensors)) if i not in self.mutable_cat_indices
        ]
        frozen_cat = {i: cat_tensors[i].squeeze(0) for i in frozen_cat_indices}
        frozen_num = [n.squeeze(0) for n in num_tensors]

        # --- Get original prediction ---
        orig_pred = self.predictor.predict(cat_tensors, num_tensors)
        orig_probs = self.predictor.predict_proba(cat_tensors, num_tensors)
        orig_prob = float(orig_probs[orig_pred])

        target_class_resolved = target_class

        if cfg.verbose:
            mut_names = [f"cat[{i}]" for i in self.mutable_cat_indices]
            print(f"Original: {self.activity_names[orig_pred]} (p={orig_prob:.3f}), "
                  f"prefix_len={prefix_len}, mutable={mut_names}")

        # --- Precompute which constraints the original activity seq satisfies ---
        n_total_cat = len(cat_tensors)
        original_cat_data = self._build_cat_data(
            original, frozen_cat, pad_len, prefix_len, n_total_cat)
        original_satisfied = {
            c for c in self.prefix_safe_constraints
            if DeclareConstraintChecker.check_with_data(
                original_acts, c, self.data_conditions.get(c), original_cat_data)
        }

        # --- GA loop ---
        population = self._initialize_population(original, prefix_len)
        all_valid_cfs: List[GACounterfactualResult] = []
        n_evaluated = 0

        for gen in range(cfg.n_generations):
            fitness_scores, gen_results = self._evaluate_fitness(
                population, original, frozen_cat, frozen_num,
                seq_len, pad_len, orig_pred, target_class_resolved, gen,
                n_total_cat,
            )
            n_evaluated += len(population)

            for res in gen_results:
                if res is not None:
                    all_valid_cfs.append(res)

            best_fitness = min(fitness_scores)

            if cfg.verbose and (gen % 10 == 0 or gen == cfg.n_generations - 1):
                n_valid = sum(1 for r in gen_results if r is not None)
                # Show best p_target for diagnostics
                best_idx = int(np.argmin(fitness_scores))
                print(f"  Gen {gen:3d}: best_fit={best_fitness:.4f}, "
                      f"valid={n_valid}/{len(population)}, "
                      f"total_valid={len(all_valid_cfs)}")

            # Elitism
            sorted_indices = np.argsort(fitness_scores)
            elite = [_clone_individual(population[i])
                     for i in sorted_indices[:cfg.elite_size]]

            # Selection + crossover + mutation
            new_population = list(elite)
            while len(new_population) < cfg.population_size:
                p_a = self._select_parent(population, fitness_scores)
                p_b = self._select_parent(population, fitness_scores)

                if random.random() < cfg.crossover_rate:
                    child = self._crossover(
                        p_a, p_b, original, original_satisfied,
                        frozen_cat, pad_len, prefix_len, n_total_cat)
                else:
                    child = _clone_individual(p_a)

                child = self._mutate(
                    child, original, original_satisfied,
                    frozen_cat, pad_len, prefix_len, n_total_cat)
                new_population.append(child)

            population = new_population[:cfg.population_size]

        diverse_cfs = self._select_diverse(all_valid_cfs, cfg.top_k)
        elapsed = time.time() - t_start

        return GACounterfactualExplanation(
            original_activity_sequence=original_acts,
            original_prediction=orig_pred,
            original_prediction_name=self.activity_names[orig_pred],
            original_probability=orig_prob,
            prefix_len=prefix_len,
            counterfactuals=diverse_cfs,
            n_generations_run=cfg.n_generations,
            n_candidates_evaluated=n_evaluated,
            n_valid_counterfactuals=len(all_valid_cfs),
            search_time_seconds=elapsed,
            n_constraints=len(self.prefix_safe_constraints),
            mutable_cat_indices=self.mutable_cat_indices,
        )

    # =========================================================================
    # Multi-channel data for MP-Declare data conditions
    # =========================================================================

    def _build_cat_data(
        self,
        individual: Individual,
        frozen_cat: Dict[int, Any],
        pad_len: int,
        prefix_len: int,
        n_total_cat: int,
    ) -> Dict[int, List[int]]:
        """Build ``{cat_feature_idx: [value_at_pos_0, …]}`` for constraint checking."""
        cat_data: Dict[int, List[int]] = {}
        for ch_pos, ci in enumerate(self.mutable_cat_indices):
            cat_data[ci] = list(individual[ch_pos])
        for ci in range(n_total_cat):
            if ci not in cat_data and ci in frozen_cat:
                t = frozen_cat[ci]
                cat_data[ci] = [int(t[pad_len + p].item()) for p in range(prefix_len)]
        return cat_data

    # =========================================================================
    # Case-Level Enforcement
    # =========================================================================

    def _enforce_case_level(self, individual: Individual) -> Individual:
        """For case-level channels, set all positions to the same value."""
        for ch in self.case_level_channels:
            # Use position 0 as the representative value
            val = individual[ch][0]
            for pos in range(len(individual[ch])):
                individual[ch][pos] = val
        return individual

    # =========================================================================
    # Population Initialization
    # =========================================================================

    def _initialize_population(
        self, original: Individual, prefix_len: int,
    ) -> List[Individual]:
        """Create initial population. Mutates all mutable channels."""
        cfg = self.config
        population: List[Individual] = []
        n_channels = len(self.mutable_cat_indices)
        half = cfg.population_size // 2

        # Half: single-position mutations (one random channel + position)
        for _ in range(half):
            ind = _clone_individual(original)
            ch = random.randint(0, n_channels - 1)
            ci = self.mutable_cat_indices[ch]
            if ch in self.case_level_channels:
                # Mutate all positions uniformly
                val = random.choice(self.valid_values[ci])
                ind[ch] = [val] * prefix_len
            else:
                pos = random.randint(0, prefix_len - 1)
                ind[ch][pos] = random.choice(self.valid_values[ci])
            population.append(ind)

        # Half: multi-position mutations across channels
        for _ in range(cfg.population_size - half):
            ind = _clone_individual(original)
            n_mutations = random.randint(2, min(3 * n_channels, prefix_len * n_channels))
            for _ in range(n_mutations):
                ch = random.randint(0, n_channels - 1)
                ci = self.mutable_cat_indices[ch]
                if ch in self.case_level_channels:
                    val = random.choice(self.valid_values[ci])
                    ind[ch] = [val] * prefix_len
                else:
                    pos = random.randint(0, prefix_len - 1)
                    ind[ch][pos] = random.choice(self.valid_values[ci])
            population.append(ind)

        return population

    # =========================================================================
    # Fitness Evaluation
    # =========================================================================

    def _evaluate_fitness(
        self,
        population: List[Individual],
        original: Individual,
        frozen_cat: Dict[int, torch.Tensor],
        frozen_num: List[torch.Tensor],
        seq_len: int,
        pad_len: int,
        orig_pred: int,
        target_class: Optional[int],
        generation: int,
        n_total_cat: int,
    ) -> Tuple[List[float], List[Optional[GACounterfactualResult]]]:
        """Evaluate fitness for all individuals. Returns (scores, results_or_None)."""
        cfg = self.config
        N = len(population)
        prefix_len = seq_len - pad_len

        # --- Build batch tensors ---
        # For each cat index, either use the individual's channel or the frozen tensor
        cat_batch: List[torch.Tensor] = []
        # Map: cat index -> batch tensor [N, seq_len]
        ind_batch_tensors: Dict[int, torch.Tensor] = {}

        for ch_pos, ci in enumerate(self.mutable_cat_indices):
            t = torch.zeros(N, seq_len, dtype=torch.long)
            for i, ind in enumerate(population):
                t[i, pad_len:] = torch.tensor(ind[ch_pos], dtype=torch.long)
            ind_batch_tensors[ci] = t

        for ci in range(n_total_cat):
            if ci in ind_batch_tensors:
                cat_batch.append(ind_batch_tensors[ci])
            else:
                cat_batch.append(
                    frozen_cat[ci].unsqueeze(0).expand(N, -1).clone()
                )

        # Num tensors
        if frozen_num:
            num_stacked = torch.stack(frozen_num, dim=-1).unsqueeze(0).expand(N, -1, -1).clone()
        else:
            num_stacked = None

        # --- Batch prediction ---
        preds, probs = self.predictor.predict_batch(cat_batch, num_stacked)

        # --- Compute fitness ---
        fitness_scores: List[float] = []
        results: List[Optional[GACounterfactualResult]] = []
        act_ch = self.activity_channel

        for i, ind in enumerate(population):
            pred_i = int(preds[i])
            probs_i = probs[i]
            ind_acts = ind[act_ch]

            # o1: validity
            if target_class is not None:
                flipped = pred_i == target_class
                p_target = float(probs_i[target_class])
            else:
                flipped = pred_i != orig_pred
                p_target = float(max(
                    probs_i[j] for j in range(len(probs_i)) if j != orig_pred
                ))
            o1 = 0.0 if flipped else (1.0 - p_target)

            # o2: proximity — total changed positions across all mutable channels
            total_diff = 0
            total_positions = 0
            for ch_pos in range(len(self.mutable_cat_indices)):
                for a, b in zip(original[ch_pos], ind[ch_pos]):
                    if a != b:
                        total_diff += 1
                total_positions += prefix_len
            o2 = total_diff / total_positions if total_positions > 0 else 0.0

            # o3: sparsity — number of (position, channel) pairs changed / total
            o3 = total_diff / total_positions if total_positions > 0 else 0.0

            # o4: plausibility — min distance to training sequences (activity channel only)
            o4 = self._compute_plausibility(ind_acts)

            # o5: conformance (activity channel + data conditions)
            if self.data_conditions:
                ind_cat_data = self._build_cat_data(
                    ind, frozen_cat, pad_len, prefix_len, n_total_cat)
                o5 = 1.0 - DeclareConstraintChecker.satisfaction_rate_with_data(
                    ind_acts, list(self.prefix_safe_constraints),
                    self.data_conditions, ind_cat_data)
            else:
                o5 = 1.0 - DeclareConstraintChecker.satisfaction_rate(
                    ind_acts, list(self.prefix_safe_constraints)
                )

            fitness = (
                cfg.w_validity * o1
                + cfg.w_proximity * o2
                + cfg.w_sparsity * o3
                + cfg.w_plausibility * o4
                + cfg.w_conformance * o5
            )
            fitness_scores.append(fitness)

            if flipped:
                # Build full cat tensors for display
                cat_seq = [cat_batch[ci][i].unsqueeze(0) for ci in range(n_total_cat)]
                num_seq = num_stacked[i].unsqueeze(0) if num_stacked is not None else None

                result = GACounterfactualResult(
                    activity_sequence=list(ind_acts),
                    cat_sequence=cat_seq,
                    num_sequence=num_seq,
                    counterfactual_prediction=pred_i,
                    counterfactual_prediction_name=self.activity_names[pred_i],
                    counterfactual_probability=float(probs_i[pred_i]),
                    proximity=o2,
                    sparsity=total_diff,
                    plausibility=o4,
                    conformance=1.0 - o5,
                    fitness=fitness,
                    generation=generation,
                )
                results.append(result)
            else:
                results.append(None)

        return fitness_scores, results

    def _compute_plausibility(self, individual_acts: List[int]) -> float:
        """Min normalized L1 distance to any training sequence (activity only)."""
        if not self.training_sequences:
            return 0.0

        ind_len = len(individual_acts)
        min_dist = float('inf')

        for train_seq in self.training_sequences:
            max_len = max(ind_len, len(train_seq))
            if max_len == 0:
                continue
            ind_padded = individual_acts + [0] * (max_len - ind_len)
            train_padded = train_seq + [0] * (max_len - len(train_seq))
            dist = sum(1 for a, b in zip(ind_padded, train_padded) if a != b) / max_len
            if dist < min_dist:
                min_dist = dist
            if min_dist == 0.0:
                break

        return min_dist if min_dist != float('inf') else 1.0

    # =========================================================================
    # Selection
    # =========================================================================

    def _select_parent(
        self, population: List[Individual], fitness_scores: List[float],
    ) -> Individual:
        """Tournament selection."""
        indices = random.sample(
            range(len(population)),
            min(self.config.tournament_size, len(population)),
        )
        best_idx = min(indices, key=lambda i: fitness_scores[i])
        return population[best_idx]

    # =========================================================================
    # Constraint-Preserving Crossover
    # =========================================================================

    def _crossover(
        self,
        parent_a: Individual,
        parent_b: Individual,
        original: Individual,
        original_satisfied: Set[DeclareConstraint],
        frozen_cat: Optional[Dict] = None,
        pad_len: int = 0,
        prefix_len: int = 0,
        n_total_cat: int = 0,
    ) -> Individual:
        """Single-point crossover on all channels with activity constraint repair."""
        plen = len(parent_a[0])
        point = random.randint(1, plen - 1)

        child: Individual = []
        for ch in range(len(self.mutable_cat_indices)):
            if ch in self.case_level_channels:
                # Case-level: inherit one parent's uniform value
                donor = parent_a if random.random() < 0.5 else parent_b
                child.append(list(donor[ch]))
            else:
                child.append(parent_a[ch][:point] + parent_b[ch][point:])

        # Repair activity channel (with data conditions if available)
        act_ch = self.activity_channel
        child[act_ch] = self._constraint_repair(
            child[act_ch], original[act_ch], original_satisfied,
            child, frozen_cat, pad_len, prefix_len or plen, n_total_cat,
        )
        return child

    # =========================================================================
    # Constraint-Preserving Mutation
    # =========================================================================

    def _mutate(
        self,
        individual: Individual,
        original: Individual,
        original_satisfied: Set[DeclareConstraint],
        frozen_cat: Optional[Dict] = None,
        pad_len: int = 0,
        prefix_len: int = 0,
        n_total_cat: int = 0,
    ) -> Individual:
        """Per-position mutation on all channels with activity constraint repair."""
        plen = len(individual[0])
        n_channels = len(self.mutable_cat_indices)
        mutation_prob = self.config.mutation_rate / plen

        mutated = _clone_individual(individual)
        for ch in range(n_channels):
            ci = self.mutable_cat_indices[ch]
            if ch in self.case_level_channels:
                # Case-level: single coin flip to mutate all positions uniformly
                if random.random() < self.config.mutation_rate:
                    val = random.choice(self.valid_values[ci])
                    mutated[ch] = [val] * plen
            else:
                for pos in range(plen):
                    if random.random() < mutation_prob:
                        mutated[ch][pos] = random.choice(self.valid_values[ci])

        # Repair activity channel (with data conditions if available)
        act_ch = self.activity_channel
        mutated[act_ch] = self._constraint_repair(
            mutated[act_ch], original[act_ch], original_satisfied,
            mutated, frozen_cat, pad_len, prefix_len or plen, n_total_cat,
        )
        return mutated

    # =========================================================================
    # Constraint Repair (activity channel only)
    # =========================================================================

    def _constraint_repair(
        self,
        act_seq: List[int],
        original_acts: List[int],
        original_satisfied: Set[DeclareConstraint],
        individual: Optional[Individual] = None,
        frozen_cat: Optional[Dict] = None,
        pad_len: int = 0,
        prefix_len: int = 0,
        n_total_cat: int = 0,
    ) -> List[int]:
        """
        Revert activity positions that cause new constraint violations.

        For each constraint the original satisfied: if violated by act_seq,
        greedily revert changed positions until the constraint is satisfied again.
        Uses MP-Declare data conditions when available.
        """
        use_data = bool(self.data_conditions and individual is not None
                        and frozen_cat is not None and n_total_cat > 0)

        for c in original_satisfied:
            dc = self.data_conditions.get(c) if use_data else None
            if use_data and dc is not None:
                cat_data = self._build_cat_data(
                    individual, frozen_cat, pad_len, prefix_len, n_total_cat)
                if DeclareConstraintChecker.check_with_data(act_seq, c, dc, cat_data):
                    continue
            else:
                if DeclareConstraintChecker.check(act_seq, c):
                    continue

            for pos in range(len(original_acts)):
                if act_seq[pos] != original_acts[pos]:
                    act_seq[pos] = original_acts[pos]
                    if use_data and dc is not None:
                        cat_data = self._build_cat_data(
                            individual, frozen_cat, pad_len, prefix_len, n_total_cat)
                        if DeclareConstraintChecker.check_with_data(act_seq, c, dc, cat_data):
                            break
                    else:
                        if DeclareConstraintChecker.check(act_seq, c):
                            break
        return act_seq

    # =========================================================================
    # Diversity Selection
    # =========================================================================

    def _select_diverse(
        self,
        counterfactuals: List[GACounterfactualResult],
        top_k: int,
    ) -> List[GACounterfactualResult]:
        """Greedy diversity selection: pick best, then next-best sufficiently different."""
        if not counterfactuals:
            return []

        sorted_cfs = sorted(counterfactuals, key=lambda c: c.fitness)

        if len(sorted_cfs) <= top_k:
            return sorted_cfs

        selected = [sorted_cfs[0]]
        for cf in sorted_cfs[1:]:
            if len(selected) >= top_k:
                break
            is_diverse = True
            for sel in selected:
                max_len = max(len(cf.activity_sequence), len(sel.activity_sequence))
                if max_len == 0:
                    continue
                cf_padded = cf.activity_sequence + [0] * (max_len - len(cf.activity_sequence))
                sel_padded = sel.activity_sequence + [0] * (max_len - len(sel.activity_sequence))
                diff_ratio = sum(
                    1 for a, b in zip(cf_padded, sel_padded) if a != b
                ) / max_len
                if diff_ratio < self.config.diversity_threshold:
                    is_diverse = False
                    break
            if is_diverse:
                selected.append(cf)

        if len(selected) < top_k:
            for cf in sorted_cfs:
                if cf not in selected:
                    selected.append(cf)
                if len(selected) >= top_k:
                    break

        return selected


# =============================================================================
# Factory Function
# =============================================================================

def create_ga_counterfactual_for_model(
    model: torch.nn.Module,
    dataset,
    activity_names: List[str],
    config: GACounterfactualConfig,
    constraints: Set[DeclareConstraint] = None,
    cat_feature_names: Optional[List[str]] = None,
    data_conditions: Optional[Dict[DeclareConstraint, DataCondition]] = None,
) -> GACounterfactual:
    """
    Convenience factory: create GACounterfactual from a model and dataset.

    Args:
        model: Trained U-ED-LSTM model.
        dataset: EventLogDataset with items (cat_tuple, num_tuple, case_id).
        activity_names: Activity vocabulary ordered by index.
        config: Configuration for the GA.
        constraints: Pre-mined Declare constraints. Empty set if None.
        cat_feature_names: Optional list of categorical feature names (for logging).
        data_conditions: Optional MP-Declare data conditions mapping constraints
            to parsed :class:`DataCondition` objects.

    Returns:
        Configured GACounterfactual instance ready for explain().
    """
    predictor = RevisedPlusModelPredictor(
        model, suffix_step=0, activity_feature=config.activity_feature,
    )

    if constraints is None:
        constraints = set()

    # Determine mutable indices (default: activity only)
    mutable = config.mutable_cat_indices if config.mutable_cat_indices is not None else [0]
    if 0 not in mutable:
        raise ValueError("Activity (cat index 0) must always be in mutable_cat_indices")

    # Compute vocab sizes for all cat features
    n_cat = len(dataset[0][0])
    cat_max = [0] * n_cat
    for item in dataset:
        for i, cat_tensor in enumerate(item[0]):
            cat_max[i] = max(cat_max[i], int(cat_tensor.max().item()))
    cat_vocab_sizes = [m + 1 for m in cat_max]

    # Extract training activity sequences (non-padded) for plausibility
    training_sequences = []
    seen_cases = set()
    for item in dataset:
        cat_tuple, num_tuple, case_id = item
        if case_id in seen_cases:
            continue
        seen_cases.add(case_id)
        act_tensor = cat_tuple[0]
        acts = [a.item() for a in act_tensor if a.item() != 0]
        if acts:
            training_sequences.append(acts)

    if config.verbose:
        mut_info = []
        for ci in mutable:
            name = cat_feature_names[ci] if cat_feature_names and ci < len(cat_feature_names) else f"cat[{ci}]"
            mut_info.append(f"{name}(vocab={cat_vocab_sizes[ci]})")
        print(f"GA Counterfactual: {len(training_sequences)} training sequences, "
              f"{len(constraints)} constraints")
        print(f"  Mutable features: {', '.join(mut_info)}")

    return GACounterfactual(
        predictor=predictor,
        activity_names=activity_names,
        constraints=constraints,
        training_sequences=training_sequences,
        cat_vocab_sizes=cat_vocab_sizes,
        mutable_cat_indices=mutable,
        config=config,
        data_conditions=data_conditions,
    )
