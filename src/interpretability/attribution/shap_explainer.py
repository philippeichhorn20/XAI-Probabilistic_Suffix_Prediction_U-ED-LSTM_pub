"""
SHAP (SHapley Additive exPlanations) for sequence models.

Implements a sequence-aware SHAP variant that computes feature attributions
by measuring the contribution of each input feature at each timestep.

This implementation uses a sampling-based approach similar to KernelSHAP,
adapted for sequential data where features have both a feature dimension
and a temporal dimension.
"""

import torch
from torch import Tensor
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from itertools import combinations
import warnings


class SequenceSHAP:
    """
    Sequence-aware SHAP explainer.

    Computes Shapley values for each (feature, timestep) combination in the input.
    Uses a sampling-based approximation for computational efficiency.

    The Shapley value for feature i is:
    φ_i = Σ_{S ⊆ N\{i}} |S|!(|N|-|S|-1)!/|N|! * [f(S ∪ {i}) - f(S)]

    For sequences, we treat each (feature, timestep) as a separate "player".
    """

    def __init__(
        self,
        model,
        data_set_categories: Tuple,
        device: str = 'cpu'
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: The model to explain
            data_set_categories: Dataset category information
            device: Device to run computations on
        """
        self.model = model
        self.data_set_categories = data_set_categories
        self.device = device

    def compute(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        target_output: str,
        target_value: Union[int, str] = 'auto',
        suffix_step: int = 0,
        n_samples: int = 100,
        baseline: str = 'zero',
        feature_level: bool = True
    ) -> Dict[str, Tensor]:
        """
        Compute SHAP values for the input prefix.

        Args:
            prefix: Input prefix as (cat_tensors, num_tensors)
            target_output: Name of output to explain
            target_value: For categorical: class index or 'auto'
                         For numerical: 'mean'
            suffix_step: Which suffix step to explain
            n_samples: Number of samples for approximation
            baseline: Baseline strategy ('zero', 'mean')
            feature_level: If True, aggregate SHAP values per feature.
                          If False, return per (feature, timestep).

        Returns:
            Dict mapping feature names to SHAP value tensors
        """
        self.model.eval()

        # Resolve 'auto' target to specific class index BEFORE perturbations
        resolved_target = self._resolve_target(prefix, target_output, target_value, suffix_step)

        # Get baseline
        baseline_prefix = self._create_baseline(prefix, baseline)

        # Get target function with resolved target
        target_fn = self._create_target_fn(target_output, resolved_target, suffix_step)

        # Get feature info
        cat_categories, num_categories = self.data_set_categories
        n_cat_features = len(cat_categories)
        n_num_features = len(num_categories)
        n_features = n_cat_features + n_num_features

        cat_tensors, num_tensors = prefix
        seq_len = cat_tensors[0].shape[1] if cat_tensors else num_tensors[0].shape[1]

        # Compute SHAP values using sampling
        if feature_level == 'hybrid':
            # Case-level features: feature-level SHAP (one player per feature)
            # Event-level features: per-timestep SHAP (one SHAP per timestep)
            shap_values = self._compute_hybrid_shap(
                prefix, baseline_prefix, target_fn, n_samples
            )
        elif feature_level:
            # Compute SHAP values per feature (aggregated across timesteps)
            shap_values = self._compute_feature_shap(
                prefix, baseline_prefix, target_fn, n_samples
            )
        else:
            # Compute SHAP values per (feature, timestep)
            shap_values = self._compute_timestep_shap(
                prefix, baseline_prefix, target_fn, n_samples
            )

        return shap_values

    def _compute_feature_shap(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        baseline: Tuple[List[Tensor], List[Tensor]],
        target_fn: Callable,
        n_samples: int
    ) -> Dict[str, Tensor]:
        """Compute SHAP values aggregated per feature."""
        cat_tensors, num_tensors = prefix
        cat_baseline, num_baseline = baseline

        cat_categories, num_categories = self.data_set_categories
        n_features = len(cat_categories) + len(num_categories)
        seq_len = cat_tensors[0].shape[1] if cat_tensors else num_tensors[0].shape[1]

        # Initialize SHAP values
        shap_values = {}
        for name, _, _ in cat_categories:
            shap_values[name] = torch.zeros(seq_len)
        for name, _, _ in num_categories:
            shap_values[name] = torch.zeros(seq_len)

        # Get baseline and input predictions
        with torch.no_grad():
            baseline_pred = target_fn(baseline)
            input_pred = target_fn(prefix)

        # Sample coalitions and compute marginal contributions
        feature_names = [name for name, _, _ in cat_categories] + [name for name, _, _ in num_categories]

        for _ in range(n_samples):
            # Random permutation of features
            perm = np.random.permutation(n_features)

            # Build coalition incrementally
            current_prefix = self._copy_prefix(baseline)

            prev_pred = baseline_pred

            for feat_idx in perm:
                # Add this feature to the coalition
                if feat_idx < len(cat_categories):
                    # Categorical feature
                    current_prefix[0][feat_idx] = cat_tensors[feat_idx].clone()
                else:
                    # Numerical feature
                    num_idx = feat_idx - len(cat_categories)
                    current_prefix[1][num_idx] = num_tensors[num_idx].clone()

                # Compute new prediction
                with torch.no_grad():
                    new_pred = target_fn(current_prefix)

                # Marginal contribution
                contribution = (new_pred - prev_pred).item()

                # Distribute contribution across timesteps (uniform for now)
                feat_name = feature_names[feat_idx]
                shap_values[feat_name] += contribution / seq_len / n_samples

                prev_pred = new_pred

        # Scale by sequence length to get per-timestep values
        for name in shap_values:
            shap_values[name] = shap_values[name].abs()

        return shap_values

    def _detect_case_level_features(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
    ) -> Tuple[List[int], List[int]]:
        """Auto-detect which features are case-level (constant in this prefix).

        Returns:
            (case_level_indices, event_level_indices) where indices are into
            the combined [cat_features..., num_features...] list.
        """
        cat_tensors, num_tensors = prefix
        cat_categories, num_categories = self.data_set_categories
        n_cat = len(cat_categories)

        case_level = []
        event_level = []

        for i, t in enumerate(cat_tensors[:n_cat]):
            vals = t.squeeze(0) if t.dim() > 1 else t
            non_pad = vals[vals != 0]
            if len(non_pad) == 0 or len(non_pad.unique()) <= 1:
                case_level.append(i)
            else:
                event_level.append(i)

        for i, t in enumerate(num_tensors[:len(num_categories)]):
            idx = n_cat + i
            vals = t.squeeze(0) if t.dim() > 1 else t
            non_pad = vals[vals != 0]
            if len(non_pad) == 0 or len(non_pad.unique()) <= 1:
                case_level.append(idx)
            else:
                event_level.append(idx)

        return case_level, event_level

    def _compute_hybrid_shap(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        baseline: Tuple[List[Tensor], List[Tensor]],
        target_fn: Callable,
        n_samples: int,
    ) -> Dict[str, Tensor]:
        """Hybrid SHAP: feature-level for case-level features, per-timestep for event-level.

        Case-level features (constant across all positions in the prefix) are
        treated as single players — toggling them flips all positions at once.

        Event-level features get per-timestep resolution: for each timestep t,
        a separate permutation SHAP is run over the event-level features at
        position t only, keeping everything else fixed.

        Only non-padding timesteps are evaluated for the per-timestep part.
        """
        cat_tensors, num_tensors = prefix
        cat_baseline, num_baseline = baseline
        cat_categories, num_categories = self.data_set_categories
        n_cat = len(cat_categories)
        n_num = len(num_categories)
        feature_names = [n for n, _, _ in cat_categories] + [n for n, _, _ in num_categories]
        seq_len = cat_tensors[0].shape[1] if cat_tensors else num_tensors[0].shape[1]

        # Detect actual prefix length (non-padding positions)
        act_tensor = cat_tensors[0].squeeze(0) if cat_tensors else num_tensors[0].squeeze(0)
        prefix_len = int((act_tensor != 0).sum().item())
        pad_len = seq_len - prefix_len

        case_level_idx, event_level_idx = self._detect_case_level_features(prefix)

        # Initialize output
        shap_values = {name: torch.zeros(seq_len) for name in feature_names}

        # =============================================================
        # Part 1: Case-level features — feature-level permutation SHAP
        # =============================================================
        if case_level_idx:
            with torch.no_grad():
                baseline_pred = target_fn(baseline)

            for _ in range(n_samples):
                perm = np.random.permutation(len(case_level_idx))
                current = self._copy_prefix(baseline)
                # Start with all event-level features already at real values
                for ei in event_level_idx:
                    if ei < n_cat:
                        current[0][ei] = cat_tensors[ei].clone()
                    else:
                        current[1][ei - n_cat] = num_tensors[ei - n_cat].clone()

                with torch.no_grad():
                    prev_pred = target_fn(current)

                for pos_in_perm in perm:
                    fi = case_level_idx[pos_in_perm]
                    if fi < n_cat:
                        current[0][fi] = cat_tensors[fi].clone()
                    else:
                        current[1][fi - n_cat] = num_tensors[fi - n_cat].clone()

                    with torch.no_grad():
                        new_pred = target_fn(current)

                    contribution = (new_pred - prev_pred).item()
                    # Distribute uniformly across non-padding timesteps only
                    if prefix_len > 0:
                        shap_values[feature_names[fi]][pad_len:] += contribution / prefix_len / n_samples
                    prev_pred = new_pred

        # =============================================================
        # Part 2: Event-level features — per-timestep permutation SHAP
        # =============================================================
        if event_level_idx:
            n_event = len(event_level_idx)

            # Only iterate over actual (non-padding) timesteps
            for t in range(pad_len, seq_len):
                for _ in range(n_samples):
                    perm = np.random.permutation(n_event)
                    # Start from baseline with case-level features at real values
                    current = self._copy_prefix(baseline)
                    for ci in case_level_idx:
                        if ci < n_cat:
                            current[0][ci] = cat_tensors[ci].clone()
                        else:
                            current[1][ci - n_cat] = num_tensors[ci - n_cat].clone()
                    # Event-level features at all OTHER timesteps: use real values
                    for ei in event_level_idx:
                        if ei < n_cat:
                            current[0][ei] = cat_tensors[ei].clone()
                        else:
                            current[1][ei - n_cat] = num_tensors[ei - n_cat].clone()
                    # Event-level features at timestep t: start from baseline
                    for ei in event_level_idx:
                        if ei < n_cat:
                            current[0][ei][0, t] = cat_baseline[ei][0, t]
                        else:
                            current[1][ei - n_cat][0, t] = num_baseline[ei - n_cat][0, t]

                    with torch.no_grad():
                        prev_pred = target_fn(current)

                    for pos_in_perm in perm:
                        ei = event_level_idx[pos_in_perm]
                        # Switch this feature at timestep t from baseline to real
                        if ei < n_cat:
                            current[0][ei][0, t] = cat_tensors[ei][0, t]
                        else:
                            current[1][ei - n_cat][0, t] = num_tensors[ei - n_cat][0, t]

                        with torch.no_grad():
                            new_pred = target_fn(current)

                        contribution = (new_pred - prev_pred).item()
                        shap_values[feature_names[ei]][t] += contribution / n_samples
                        prev_pred = new_pred

        # Take absolute values
        for name in shap_values:
            shap_values[name] = shap_values[name].abs()

        return shap_values

    def _compute_timestep_shap(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        baseline: Tuple[List[Tensor], List[Tensor]],
        target_fn: Callable,
        n_samples: int
    ) -> Dict[str, Tensor]:
        """Compute SHAP values per (feature, timestep) pair."""
        cat_tensors, num_tensors = prefix
        cat_baseline, num_baseline = baseline

        cat_categories, num_categories = self.data_set_categories
        seq_len = cat_tensors[0].shape[1] if cat_tensors else num_tensors[0].shape[1]

        # Initialize SHAP values
        shap_values = {}
        for name, _, _ in cat_categories:
            shap_values[name] = torch.zeros(seq_len)
        for name, _, _ in num_categories:
            shap_values[name] = torch.zeros(seq_len)

        feature_names = [name for name, _, _ in cat_categories] + [name for name, _, _ in num_categories]
        n_features = len(feature_names)

        # Total number of (feature, timestep) players
        n_players = n_features * seq_len

        # For efficiency, use feature-level sampling with timestep distribution
        for _ in range(n_samples):
            # Random binary mask for each (feature, timestep)
            mask = np.random.binomial(1, 0.5, size=(n_features, seq_len))

            # Create masked prefix
            masked_prefix = self._apply_mask(prefix, baseline, mask, cat_categories, num_categories)

            with torch.no_grad():
                pred_with_mask = target_fn(masked_prefix)

            # For each feature and timestep, compute marginal contribution
            for feat_idx, feat_name in enumerate(feature_names):
                for t in range(seq_len):
                    if mask[feat_idx, t] == 1:
                        # Feature is included, compute contribution by removing it
                        mask_without = mask.copy()
                        mask_without[feat_idx, t] = 0
                        masked_without = self._apply_mask(prefix, baseline, mask_without, cat_categories, num_categories)

                        with torch.no_grad():
                            pred_without = target_fn(masked_without)

                        contribution = (pred_with_mask - pred_without).item()
                        shap_values[feat_name][t] += contribution / n_samples
                    else:
                        # Feature is excluded, compute contribution by adding it
                        mask_with = mask.copy()
                        mask_with[feat_idx, t] = 1
                        masked_with = self._apply_mask(prefix, baseline, mask_with, cat_categories, num_categories)

                        with torch.no_grad():
                            pred_with = target_fn(masked_with)

                        contribution = (pred_with - pred_with_mask).item()
                        shap_values[feat_name][t] += contribution / n_samples

        # Take absolute values for importance
        for name in shap_values:
            shap_values[name] = shap_values[name].abs()

        return shap_values

    def _apply_mask(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        baseline: Tuple[List[Tensor], List[Tensor]],
        mask: np.ndarray,
        cat_categories: List,
        num_categories: List
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Apply binary mask to create a coalition."""
        cat_tensors, num_tensors = prefix
        cat_baseline, num_baseline = baseline

        n_cat = len(cat_categories)
        n_num = len(num_categories)

        masked_cat = []
        for i in range(len(cat_tensors)):
            if i < n_cat:
                masked = cat_baseline[i].clone()
                for t in range(mask.shape[1]):
                    if mask[i, t] == 1:
                        masked[0, t] = cat_tensors[i][0, t]
                masked_cat.append(masked)
            else:
                # Features beyond data_set_categories: keep original
                masked_cat.append(cat_tensors[i].clone())

        masked_num = []
        for i in range(len(num_tensors)):
            if i < n_num:
                masked = num_baseline[i].clone()
                for t in range(mask.shape[1]):
                    if mask[n_cat + i, t] == 1:
                        masked[0, t] = num_tensors[i][0, t]
                masked_num.append(masked)
            else:
                # Features beyond data_set_categories: keep original
                masked_num.append(num_tensors[i].clone())

        return (masked_cat, masked_num)

    def _create_baseline(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        strategy: str
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Create baseline prefix."""
        cat_tensors, num_tensors = prefix

        if strategy == 'zero':
            cat_baseline = [torch.zeros_like(t) for t in cat_tensors]
            num_baseline = [torch.zeros_like(t) for t in num_tensors]
        elif strategy == 'mean':
            # Use mean of current prefix as baseline
            cat_baseline = [torch.full_like(t, t.float().mean().item()).long() for t in cat_tensors]
            num_baseline = [torch.full_like(t, t.mean().item()) for t in num_tensors]
        else:
            raise ValueError(f"Unknown baseline strategy: {strategy}")

        return (cat_baseline, num_baseline)

    def _copy_prefix(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Deep copy prefix tensors."""
        cat_tensors, num_tensors = prefix
        return (
            [t.clone() for t in cat_tensors],
            [t.clone() for t in num_tensors]
        )

    def _resolve_target(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        target_output: str,
        target_value: Union[int, str],
        suffix_step: int
    ) -> Union[int, str]:
        """
        Resolve 'auto' target to a specific class index based on original prediction.

        This ensures that attributions are computed with respect to a consistent
        class, rather than tracking whatever class has the highest logit after
        each perturbation.

        Args:
            prefix: Original input prefix
            target_output: Name of output feature
            target_value: 'auto' or specific class index
            suffix_step: Which suffix step

        Returns:
            Resolved class index (int) or 'mean' for numerical targets
        """
        if target_value != 'auto':
            return target_value

        cat_categories, num_categories = self.data_set_categories
        is_categorical = any(name == target_output for name, _, _ in cat_categories)

        if not is_categorical:
            return 'mean'

        # Get prediction for original input
        with torch.no_grad():
            predictions, _, _, _ = self.model(prefix)
            cat_preds, _ = predictions
            key = f"{target_output}_mean"
            logits = cat_preds[key]
            if logits.dim() == 3:
                logits = logits[suffix_step]
            # Return the predicted class index
            return logits.argmax(dim=-1).item()

    def _create_target_fn(
        self,
        target_output: str,
        target_value: Union[int, str],
        suffix_step: int
    ) -> Callable:
        """Create function that extracts the target output.

        Note: target_value should already be resolved (not 'auto') before calling this.
        """
        cat_categories, num_categories = self.data_set_categories

        # Determine if categorical or numerical
        is_categorical = any(name == target_output for name, _, _ in cat_categories)

        def target_fn(prefix: Tuple[List[Tensor], List[Tensor]]) -> Tensor:
            predictions, _, _, _ = self.model(prefix)
            cat_preds, num_preds = predictions

            if is_categorical:
                key = f"{target_output}_mean"
                logits = cat_preds[key]
                if logits.dim() == 3:
                    logits = logits[suffix_step]

                # target_value is already resolved to a specific class index
                return logits[0, target_value]
            else:
                key = f"{target_output}_mean"
                value = num_preds[key]
                if value.dim() == 3:
                    value = value[suffix_step]
                return value[0, 0]

        return target_fn
