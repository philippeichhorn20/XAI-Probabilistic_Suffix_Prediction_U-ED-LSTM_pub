"""
Target selectors for Integrated Gradients.

These classes define how to select the target output for computing attributions.
For multi-output models (predicting activities, resources, timestamps), we need to
specify which output to explain.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor


class TargetSelector(ABC):
    """
    Abstract base class for selecting the target output for IG computation.

    The target selector takes the model predictions and returns a scalar tensor
    that will be differentiated with respect to the inputs.
    """

    @abstractmethod
    def select(
        self,
        predictions: List[Dict[str, Tensor]],
        suffix_step: int = 0
    ) -> Tensor:
        """
        Select the target output from model predictions.

        Args:
            predictions: Model output in format:
                        [cat_predictions_dict, num_predictions_dict]
                        Each dict has keys like 'feature_name_mean', 'feature_name_var'
            suffix_step: Which step in the predicted suffix to target (0-indexed).

        Returns:
            Scalar tensor for backpropagation.
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return a human-readable description of what is being targeted."""
        pass

    def select_direct(
        self,
        predictions: Tuple[Dict[str, Tensor], Dict[str, Tensor]]
    ) -> Tensor:
        """Select from predictions without suffix dimension.

        Default implementation wraps predictions in a list and calls select with step 0.
        Subclasses may override for efficiency.
        """
        return self.select(list(predictions), suffix_step=0)


class CategoricalTargetSelector(TargetSelector):
    """
    Target selector for categorical outputs (activity, resource, etc.).

    Supports:
    - Specific class index: Explain why the model predicts class i
    - 'auto': Explain the predicted class (argmax of logits)
    - Probability of specific class vs others
    """

    def __init__(
        self,
        feature_name: str,
        target_class: Union[int, str] = 'auto',
        use_logit: bool = True,
        use_log_softmax: bool = False
    ):
        """
        Initialize categorical target selector.

        Args:
            feature_name: Name of the categorical feature (e.g., 'concept:name', 'resource')
            target_class: Class index to target, or 'auto' to use argmax prediction
            use_logit: If True, target the raw logit. If False, target softmax probability.
            use_log_softmax: If True, use log-softmax (useful for numerical stability).
                            Only applies if use_logit is False.
        """
        self.feature_name = feature_name
        self.target_class = target_class
        self.use_logit = use_logit
        self.use_log_softmax = use_log_softmax
        self._resolved_class = None  # Will be set after selection

    def select(
        self,
        predictions: List[Dict[str, Tensor]],
        suffix_step: int = 0
    ) -> Tensor:
        """Select the target logit or probability for the specified class."""
        cat_predictions = predictions[0]

        # Get the mean predictions (logits) for the feature
        key = f"{self.feature_name}_mean"
        if key not in cat_predictions:
            available = list(cat_predictions.keys())
            raise KeyError(
                f"Feature '{self.feature_name}' not found. "
                f"Available categorical features: {available}"
            )

        logits = cat_predictions[key]  # Shape: [suffix_len, batch, num_classes] or [batch, num_classes]

        # Handle suffix step indexing
        if logits.dim() == 3:
            logits = logits[suffix_step]  # Shape: [batch, num_classes]

        return self._select_from_logits(logits)

    def select_direct(
        self,
        predictions: Tuple[Dict[str, Tensor], Dict[str, Tensor]]
    ) -> Tensor:
        """Select from predictions without suffix dimension (direct output)."""
        cat_predictions = predictions[0]
        key = f"{self.feature_name}_mean"
        logits = cat_predictions[key]  # Shape: [batch, num_classes]
        return self._select_from_logits(logits)

    def _select_from_logits(self, logits: Tensor) -> Tensor:
        """Internal method to select target from logits tensor."""
        # Determine target class
        # IMPORTANT: Only resolve 'auto' once on the first call, then use cached value.
        # This ensures consistent target during IG integration path.
        if self.target_class == 'auto':
            if self._resolved_class is None:
                # First call: resolve based on current (original input) prediction
                with torch.no_grad():
                    self._resolved_class = logits.argmax(dim=-1).item()
            target_idx = self._resolved_class
        else:
            target_idx = self.target_class
            self._resolved_class = target_idx

        # Extract target value
        if self.use_logit:
            # Target raw logit
            target = logits[0, target_idx]  # Assuming batch size 1
        else:
            if self.use_log_softmax:
                probs = torch.log_softmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
            target = probs[0, target_idx]

        return target

    def get_description(self) -> str:
        class_str = (
            f"class {self._resolved_class} (auto)"
            if self.target_class == 'auto'
            else f"class {self.target_class}"
        )
        output_type = "logit" if self.use_logit else "probability"
        return f"{self.feature_name} {output_type} for {class_str}"

    @property
    def resolved_class(self) -> Optional[int]:
        """Return the resolved class index after selection."""
        return self._resolved_class


class NumericalTargetSelector(TargetSelector):
    """
    Target selector for numerical outputs (timestamps, durations, etc.).

    Supports targeting:
    - Mean prediction
    - Variance prediction (for uncertainty analysis)
    """

    def __init__(
        self,
        feature_name: str,
        target_type: str = 'mean'
    ):
        """
        Initialize numerical target selector.

        Args:
            feature_name: Name of the numerical feature
                         (e.g., 'case_elapsed_time', 'time_since_last_event')
            target_type: 'mean' or 'var' to target mean prediction or variance
        """
        self.feature_name = feature_name
        self.target_type = target_type

    def select(
        self,
        predictions: List[Dict[str, Tensor]],
        suffix_step: int = 0
    ) -> Tensor:
        """Select the target numerical prediction."""
        num_predictions = predictions[1]

        key = f"{self.feature_name}_{self.target_type}"
        if key not in num_predictions:
            available = list(num_predictions.keys())
            raise KeyError(
                f"Feature '{self.feature_name}_{self.target_type}' not found. "
                f"Available numerical features: {available}"
            )

        value = num_predictions[key]  # Shape: [suffix_len, batch, 1] or [batch, 1]

        # Handle suffix step indexing
        if value.dim() == 3:
            value = value[suffix_step]  # Shape: [batch, 1]

        # Return scalar (assuming batch size 1)
        return value[0, 0]

    def get_description(self) -> str:
        return f"{self.feature_name} {self.target_type} prediction"


class MultiTargetSelector(TargetSelector):
    """
    Combine multiple target selectors with weighted sum.

    Useful for understanding joint predictions or comparing attributions.
    """

    def __init__(
        self,
        selectors: List[TargetSelector],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize multi-target selector.

        Args:
            selectors: List of target selectors to combine
            weights: Weights for each selector. Defaults to equal weights.
        """
        self.selectors = selectors
        self.weights = weights or [1.0] * len(selectors)
        assert len(self.selectors) == len(self.weights)

    def select(
        self,
        predictions: List[Dict[str, Tensor]],
        suffix_step: int = 0
    ) -> Tensor:
        """Select weighted sum of targets."""
        total = 0.0
        for selector, weight in zip(self.selectors, self.weights):
            target = selector.select(predictions, suffix_step)
            total = total + weight * target
        return total

    def get_description(self) -> str:
        parts = []
        for selector, weight in zip(self.selectors, self.weights):
            parts.append(f"{weight}*({selector.get_description()})")
        return " + ".join(parts)


class DifferenceTargetSelector(TargetSelector):
    """
    Target the difference between two predictions.

    Useful for understanding what makes the model prefer one class over another.
    """

    def __init__(
        self,
        selector_a: TargetSelector,
        selector_b: TargetSelector
    ):
        """
        Initialize difference selector.

        Args:
            selector_a: First target selector
            selector_b: Second target selector (will be subtracted from first)
        """
        self.selector_a = selector_a
        self.selector_b = selector_b

    def select(
        self,
        predictions: List[Dict[str, Tensor]],
        suffix_step: int = 0
    ) -> Tensor:
        """Select difference between two targets."""
        a = self.selector_a.select(predictions, suffix_step)
        b = self.selector_b.select(predictions, suffix_step)
        return a - b

    def get_description(self) -> str:
        return f"({self.selector_a.get_description()}) - ({self.selector_b.get_description()})"


def create_target_selector(
    target_output: str,
    target_value: Union[int, str] = 'auto',
    feature_type: str = 'categorical',
    **kwargs
) -> TargetSelector:
    """
    Factory function to create target selectors.

    Args:
        target_output: Name of the output feature (e.g., 'concept:name', 'resource',
                      'case_elapsed_time')
        target_value: For categorical: class index or 'auto'.
                     For numerical: 'mean' or 'var'.
        feature_type: 'categorical' or 'numerical'
        **kwargs: Additional arguments passed to the selector

    Returns:
        Appropriate TargetSelector instance
    """
    if feature_type == 'categorical':
        return CategoricalTargetSelector(
            feature_name=target_output,
            target_class=target_value,
            **kwargs
        )
    elif feature_type == 'numerical':
        return NumericalTargetSelector(
            feature_name=target_output,
            target_type=target_value if isinstance(target_value, str) else 'mean',
            **kwargs
        )
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
