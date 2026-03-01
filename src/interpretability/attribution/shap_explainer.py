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
        if feature_level:
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
