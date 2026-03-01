"""
LIME (Local Interpretable Model-agnostic Explanations) for sequence models.

Implements a sequence-aware LIME variant that computes feature attributions
by fitting a local linear model around the input.

This implementation adapts LIME for sequential data where features have both
a feature dimension and a temporal dimension.

Reference: Ribeiro et al., "Why Should I Trust You?": Explaining the Predictions
of Any Classifier (KDD 2016)
"""

import torch
from torch import Tensor
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import Ridge


class SequenceLIME:
    """
    Sequence-aware LIME explainer.

    Computes local feature importances for each (feature, timestep) combination
    by fitting a weighted linear model on perturbed samples.

    The algorithm:
    1. Generate binary perturbation masks for (feature, timestep) pairs
    2. Create perturbed inputs by replacing masked positions with baseline values
    3. Get model predictions for all perturbed inputs
    4. Weight samples by proximity to original (exponential kernel)
    5. Fit weighted Ridge regression: prediction ~ mask
    6. Return regression coefficients as feature importances
    """

    def __init__(
        self,
        model,
        data_set_categories: Tuple,
        device: str = 'cpu'
    ):
        """
        Initialize LIME explainer.

        Args:
            model: The model to explain
            data_set_categories: Dataset category information as tuple of
                                 (cat_categories, num_categories)
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
        n_samples: int = 500,
        kernel_width: float = 0.25,
        baseline: str = 'zero',
        feature_level: bool = True
    ) -> Dict[str, Tensor]:
        """
        Compute LIME feature importances for the input prefix.

        Args:
            prefix: Input prefix as (cat_tensors, num_tensors)
            target_output: Name of output to explain (e.g., 'Activity')
            target_value: For categorical: class index or 'auto' (uses predicted class)
                         For numerical: 'mean'
            suffix_step: Which suffix step to explain (0 = next event)
            n_samples: Number of perturbed samples to generate
            kernel_width: Width of exponential kernel (smaller = more local)
            baseline: Baseline strategy ('zero', 'mean')
            feature_level: If True, aggregate importances per feature.
                          If False, return per (feature, timestep).

        Returns:
            Dict mapping feature names to importance tensors
        """
        self.model.eval()

        # Resolve 'auto' target to specific class index BEFORE perturbations
        resolved_target = self._resolve_target(prefix, target_output, target_value, suffix_step)

        # Get baseline prefix
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

        # Compute LIME importances
        if feature_level:
            # Compute importances per feature (aggregated across timesteps)
            importances = self._compute_feature_lime(
                prefix, baseline_prefix, target_fn, n_samples, kernel_width
            )
        else:
            # Compute importances per (feature, timestep)
            importances = self._compute_timestep_lime(
                prefix, baseline_prefix, target_fn, n_samples, kernel_width
            )

        return importances

    def _compute_feature_lime(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        baseline: Tuple[List[Tensor], List[Tensor]],
        target_fn: Callable,
        n_samples: int,
        kernel_width: float
    ) -> Dict[str, Tensor]:
        """Compute LIME importances aggregated per feature."""
        cat_tensors, num_tensors = prefix
        cat_categories, num_categories = self.data_set_categories
        n_features = len(cat_categories) + len(num_categories)
        seq_len = cat_tensors[0].shape[1] if cat_tensors else num_tensors[0].shape[1]

        feature_names = [name for name, _, _ in cat_categories] + [name for name, _, _ in num_categories]

        # Generate perturbation masks (binary: 1 = use original, 0 = use baseline)
        # Shape: (n_samples, n_features)
        masks = np.random.binomial(1, 0.5, size=(n_samples, n_features))

        # Always include the original (all 1s) and baseline (all 0s)
        masks[0] = np.ones(n_features)
        if n_samples > 1:
            masks[1] = np.zeros(n_features)

        # Get predictions for all perturbed samples
        predictions = []
        for mask in masks:
            perturbed = self._apply_feature_mask(prefix, baseline, mask, cat_categories, num_categories)
            with torch.no_grad():
                pred = target_fn(perturbed)
            predictions.append(pred.item())

        predictions = np.array(predictions)

        # Compute kernel weights (proximity to original)
        # Distance = fraction of features that differ from original
        distances = 1.0 - masks.mean(axis=1)  # 0 for original, 1 for baseline
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        # Fit weighted Ridge regression
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(masks, predictions, sample_weight=weights)

        # Extract coefficients as feature importances
        coefficients = ridge.coef_

        # Build result dict
        importances = {}
        for i, name in enumerate(feature_names):
            # Return absolute importance, distributed uniformly across timesteps
            importances[name] = torch.full((seq_len,), abs(coefficients[i]) / seq_len)

        return importances

    def _compute_timestep_lime(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        baseline: Tuple[List[Tensor], List[Tensor]],
        target_fn: Callable,
        n_samples: int,
        kernel_width: float
    ) -> Dict[str, Tensor]:
        """Compute LIME importances per (feature, timestep) pair."""
        cat_tensors, num_tensors = prefix
        cat_categories, num_categories = self.data_set_categories
        n_features = len(cat_categories) + len(num_categories)
        seq_len = cat_tensors[0].shape[1] if cat_tensors else num_tensors[0].shape[1]

        feature_names = [name for name, _, _ in cat_categories] + [name for name, _, _ in num_categories]

        # Total number of interpretable features
        n_interp_features = n_features * seq_len

        # Generate perturbation masks
        # Shape: (n_samples, n_features, seq_len)
        masks = np.random.binomial(1, 0.5, size=(n_samples, n_features, seq_len))

        # Always include original and baseline
        masks[0] = np.ones((n_features, seq_len))
        if n_samples > 1:
            masks[1] = np.zeros((n_features, seq_len))

        # Get predictions for all perturbed samples
        predictions = []
        for mask in masks:
            perturbed = self._apply_timestep_mask(prefix, baseline, mask, cat_categories, num_categories)
            with torch.no_grad():
                pred = target_fn(perturbed)
            predictions.append(pred.item())

        predictions = np.array(predictions)

        # Flatten masks for regression
        masks_flat = masks.reshape(n_samples, -1)

        # Compute kernel weights
        distances = 1.0 - masks_flat.mean(axis=1)
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        # Fit weighted Ridge regression
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(masks_flat, predictions, sample_weight=weights)

        # Extract coefficients and reshape
        coefficients = ridge.coef_.reshape(n_features, seq_len)

        # Build result dict
        importances = {}
        for i, name in enumerate(feature_names):
            importances[name] = torch.tensor(np.abs(coefficients[i]), dtype=torch.float32)

        return importances

    def _apply_feature_mask(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        baseline: Tuple[List[Tensor], List[Tensor]],
        mask: np.ndarray,
        cat_categories: List,
        num_categories: List
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Apply feature-level binary mask to create perturbed input."""
        cat_tensors, num_tensors = prefix
        cat_baseline, num_baseline = baseline
        n_cat = len(cat_categories)
        n_num = len(num_categories)

        # Only mask the features defined in data_set_categories
        masked_cat = []
        for i in range(len(cat_tensors)):
            if i < n_cat and mask[i] == 1:
                masked_cat.append(cat_tensors[i].clone())
            elif i < n_cat:
                masked_cat.append(cat_baseline[i].clone())
            else:
                # Features beyond data_set_categories: keep original
                masked_cat.append(cat_tensors[i].clone())

        masked_num = []
        for i in range(len(num_tensors)):
            if i < n_num and mask[n_cat + i] == 1:
                masked_num.append(num_tensors[i].clone())
            elif i < n_num:
                masked_num.append(num_baseline[i].clone())
            else:
                # Features beyond data_set_categories: keep original
                masked_num.append(num_tensors[i].clone())

        return (masked_cat, masked_num)

    def _apply_timestep_mask(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        baseline: Tuple[List[Tensor], List[Tensor]],
        mask: np.ndarray,
        cat_categories: List,
        num_categories: List
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Apply (feature, timestep) binary mask to create perturbed input."""
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
        """Create baseline prefix for perturbations."""
        cat_tensors, num_tensors = prefix

        if strategy == 'zero':
            cat_baseline = [torch.zeros_like(t) for t in cat_tensors]
            num_baseline = [torch.zeros_like(t) for t in num_tensors]
        elif strategy == 'mean':
            cat_baseline = [torch.full_like(t, t.float().mean().item()).long() for t in cat_tensors]
            num_baseline = [torch.full_like(t, t.mean().item()) for t in num_tensors]
        else:
            raise ValueError(f"Unknown baseline strategy: {strategy}")

        return (cat_baseline, num_baseline)

    def _resolve_target(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        target_output: str,
        target_value: Union[int, str],
        suffix_step: int
    ) -> Union[int, str]:
        """
        Resolve 'auto' target to a specific class index based on original prediction.

        This ensures attributions are computed with respect to a consistent class.
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
            return logits.argmax(dim=-1).item()

    def _create_target_fn(
        self,
        target_output: str,
        target_value: Union[int, str],
        suffix_step: int
    ) -> Callable:
        """Create function that extracts the target output value."""
        cat_categories, num_categories = self.data_set_categories
        is_categorical = any(name == target_output for name, _, _ in cat_categories)

        def target_fn(prefix: Tuple[List[Tensor], List[Tensor]]) -> Tensor:
            predictions, _, _, _ = self.model(prefix)
            cat_preds, num_preds = predictions

            if is_categorical:
                key = f"{target_output}_mean"
                logits = cat_preds[key]
                if logits.dim() == 3:
                    logits = logits[suffix_step]
                # Return logit for target class (or probability after softmax)
                return logits[0, target_value]
            else:
                key = f"{target_output}_mean"
                value = num_preds[key]
                if value.dim() == 3:
                    value = value[suffix_step]
                return value[0, 0]

        return target_fn
