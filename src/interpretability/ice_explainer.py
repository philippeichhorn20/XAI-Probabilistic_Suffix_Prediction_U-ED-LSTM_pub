"""
ICE (Individual Conditional Expectation) for sequence models.

ICE plots show how the model's prediction changes as we vary a single feature
while keeping all other features fixed. This helps understand the marginal
effect of each feature on the prediction.

For sequences, we can:
1. Vary a feature at a specific timestep
2. Vary a feature across all timesteps simultaneously
3. Compute sensitivity (derivative of prediction w.r.t. feature)
"""

import torch
from torch import Tensor
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class SequenceICE:
    """
    ICE (Individual Conditional Expectation) explainer for sequences.

    Computes how the prediction changes when varying individual features,
    which helps understand feature sensitivity and importance.
    """

    def __init__(
        self,
        model,
        data_set_categories: Tuple,
        device: str = 'cpu'
    ):
        """
        Initialize ICE explainer.

        Args:
            model: The model to explain
            data_set_categories: Dataset category information
            device: Device to run computations on
        """
        self.model = model
        self.data_set_categories = data_set_categories
        self.device = device

    def compute_sensitivity(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        target_output: str,
        target_value: Union[int, str] = 'auto',
        suffix_step: int = 0,
        perturbation_size: float = 0.1
    ) -> Dict[str, Tensor]:
        """
        Compute feature sensitivity using finite differences.

        For each feature at each timestep, measures how much the prediction
        changes when the feature is perturbed.

        Args:
            prefix: Input prefix as (cat_tensors, num_tensors)
            target_output: Name of output to explain
            target_value: For categorical: class index or 'auto'
            suffix_step: Which suffix step to explain
            perturbation_size: Size of perturbation for numerical features

        Returns:
            Dict mapping feature names to sensitivity tensors [seq_len]
        """
        self.model.eval()

        cat_tensors, num_tensors = prefix
        cat_categories, num_categories = self.data_set_categories
        seq_len = cat_tensors[0].shape[1] if cat_tensors else num_tensors[0].shape[1]

        # Resolve 'auto' target to specific class index BEFORE perturbations
        resolved_target = self._resolve_target(prefix, target_output, target_value, suffix_step)

        # Get target function with resolved target
        target_fn = self._create_target_fn(target_output, resolved_target, suffix_step)

        # Get baseline prediction
        with torch.no_grad():
            baseline_pred = target_fn(prefix).item()

        sensitivities = {}

        # Compute sensitivity for categorical features (using leave-one-out)
        for feat_idx, (name, num_classes, _) in enumerate(cat_categories):
            sensitivity = torch.zeros(seq_len)

            for t in range(seq_len):
                original_value = cat_tensors[feat_idx][0, t].item()

                # Try setting to zero (baseline)
                perturbed = self._perturb_categorical(prefix, feat_idx, t, 0)
                with torch.no_grad():
                    perturbed_pred = target_fn(perturbed).item()

                sensitivity[t] = abs(baseline_pred - perturbed_pred)

            sensitivities[name] = sensitivity

        # Compute sensitivity for numerical features (using finite differences)
        for feat_idx, (name, _, _) in enumerate(num_categories):
            sensitivity = torch.zeros(seq_len)

            for t in range(seq_len):
                original_value = num_tensors[feat_idx][0, t].item()

                # Positive perturbation
                perturbed_pos = self._perturb_numerical(
                    prefix, feat_idx, t, original_value + perturbation_size
                )
                with torch.no_grad():
                    pred_pos = target_fn(perturbed_pos).item()

                # Negative perturbation
                perturbed_neg = self._perturb_numerical(
                    prefix, feat_idx, t, original_value - perturbation_size
                )
                with torch.no_grad():
                    pred_neg = target_fn(perturbed_neg).item()

                # Central difference approximation of derivative
                sensitivity[t] = abs(pred_pos - pred_neg) / (2 * perturbation_size)

            sensitivities[name] = sensitivity

        return sensitivities

    def compute_ice_curves(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        feature_name: str,
        timestep: int,
        target_output: str,
        target_value: Union[int, str] = 'auto',
        suffix_step: int = 0,
        n_points: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ICE curve for a specific feature at a specific timestep.

        Shows how the prediction changes as the feature value varies.

        Args:
            prefix: Input prefix
            feature_name: Name of feature to vary
            timestep: Which timestep to vary
            target_output: Name of output to explain
            target_value: Target class or 'auto'
            suffix_step: Which suffix step
            n_points: Number of points in the ICE curve

        Returns:
            Tuple of (feature_values, predictions) arrays
        """
        self.model.eval()

        cat_categories, num_categories = self.data_set_categories

        # Resolve 'auto' target to specific class index BEFORE perturbations
        resolved_target = self._resolve_target(prefix, target_output, target_value, suffix_step)
        target_fn = self._create_target_fn(target_output, resolved_target, suffix_step)

        # Find the feature
        is_categorical = False
        feat_idx = None

        for idx, (name, num_classes, _) in enumerate(cat_categories):
            if name == feature_name:
                is_categorical = True
                feat_idx = idx
                n_classes = num_classes
                break

        if feat_idx is None:
            for idx, (name, _, _) in enumerate(num_categories):
                if name == feature_name:
                    feat_idx = idx
                    break

        if feat_idx is None:
            raise ValueError(f"Feature '{feature_name}' not found")

        if is_categorical:
            # For categorical, try all possible values
            feature_values = np.arange(n_classes)
            predictions = []

            for val in feature_values:
                perturbed = self._perturb_categorical(prefix, feat_idx, timestep, int(val))
                with torch.no_grad():
                    pred = target_fn(perturbed).item()
                predictions.append(pred)

            return feature_values, np.array(predictions)
        else:
            # For numerical, create a range around the current value
            cat_tensors, num_tensors = prefix
            current_value = num_tensors[feat_idx][0, timestep].item()

            # Create range centered at current value
            range_size = 2.0  # +/- 1 standard deviation-ish
            feature_values = np.linspace(
                current_value - range_size,
                current_value + range_size,
                n_points
            )

            predictions = []
            for val in feature_values:
                perturbed = self._perturb_numerical(prefix, feat_idx, timestep, val)
                with torch.no_grad():
                    pred = target_fn(perturbed).item()
                predictions.append(pred)

            return feature_values, np.array(predictions)

    def compute_feature_importance(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        target_output: str,
        target_value: Union[int, str] = 'auto',
        suffix_step: int = 0,
        method: str = 'permutation',
        n_permutations: int = 10
    ) -> Dict[str, Tensor]:
        """
        Compute feature importance using permutation or ablation.

        Args:
            prefix: Input prefix
            target_output: Name of output to explain
            target_value: Target class or 'auto'
            suffix_step: Which suffix step
            method: 'permutation' or 'ablation'
            n_permutations: Number of permutations (for permutation method)

        Returns:
            Dict mapping feature names to importance tensors [seq_len]
        """
        self.model.eval()

        cat_tensors, num_tensors = prefix
        cat_categories, num_categories = self.data_set_categories
        seq_len = cat_tensors[0].shape[1] if cat_tensors else num_tensors[0].shape[1]

        # Resolve 'auto' target to specific class index BEFORE perturbations
        resolved_target = self._resolve_target(prefix, target_output, target_value, suffix_step)
        target_fn = self._create_target_fn(target_output, resolved_target, suffix_step)

        # Get baseline prediction
        with torch.no_grad():
            baseline_pred = target_fn(prefix).item()

        importances = {}

        if method == 'ablation':
            # Set feature to zero and measure change
            for feat_idx, (name, _, _) in enumerate(cat_categories):
                importance = torch.zeros(seq_len)
                for t in range(seq_len):
                    ablated = self._perturb_categorical(prefix, feat_idx, t, 0)
                    with torch.no_grad():
                        ablated_pred = target_fn(ablated).item()
                    importance[t] = abs(baseline_pred - ablated_pred)
                importances[name] = importance

            for feat_idx, (name, _, _) in enumerate(num_categories):
                importance = torch.zeros(seq_len)
                for t in range(seq_len):
                    ablated = self._perturb_numerical(prefix, feat_idx, t, 0.0)
                    with torch.no_grad():
                        ablated_pred = target_fn(ablated).item()
                    importance[t] = abs(baseline_pred - ablated_pred)
                importances[name] = importance

        elif method == 'permutation':
            # Permute feature values and measure change
            for feat_idx, (name, _, _) in enumerate(cat_categories):
                importance = torch.zeros(seq_len)
                for t in range(seq_len):
                    perm_diffs = []
                    for _ in range(n_permutations):
                        # Random value from the sequence
                        random_t = np.random.randint(0, seq_len)
                        random_val = cat_tensors[feat_idx][0, random_t].item()
                        perturbed = self._perturb_categorical(prefix, feat_idx, t, int(random_val))
                        with torch.no_grad():
                            perturbed_pred = target_fn(perturbed).item()
                        perm_diffs.append(abs(baseline_pred - perturbed_pred))
                    importance[t] = np.mean(perm_diffs)
                importances[name] = importance

            for feat_idx, (name, _, _) in enumerate(num_categories):
                importance = torch.zeros(seq_len)
                for t in range(seq_len):
                    perm_diffs = []
                    for _ in range(n_permutations):
                        random_t = np.random.randint(0, seq_len)
                        random_val = num_tensors[feat_idx][0, random_t].item()
                        perturbed = self._perturb_numerical(prefix, feat_idx, t, random_val)
                        with torch.no_grad():
                            perturbed_pred = target_fn(perturbed).item()
                        perm_diffs.append(abs(baseline_pred - perturbed_pred))
                    importance[t] = np.mean(perm_diffs)
                importances[name] = importance

        else:
            raise ValueError(f"Unknown method: {method}")

        return importances

    def _perturb_categorical(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        feat_idx: int,
        timestep: int,
        new_value: int
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Create a copy of prefix with one categorical value changed."""
        cat_tensors, num_tensors = prefix
        new_cat = [t.clone() for t in cat_tensors]
        new_cat[feat_idx][0, timestep] = new_value
        return (new_cat, [t.clone() for t in num_tensors])

    def _perturb_numerical(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        feat_idx: int,
        timestep: int,
        new_value: float
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Create a copy of prefix with one numerical value changed."""
        cat_tensors, num_tensors = prefix
        new_num = [t.clone() for t in num_tensors]
        new_num[feat_idx][0, timestep] = new_value
        return ([t.clone() for t in cat_tensors], new_num)

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
