"""
Base classes and interfaces for perturbation-based explainability methods.

This module defines abstract base classes that all counterfactual explanation
methods should implement, ensuring a consistent interface across different algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Generic, TypeVar
import numpy as np
import torch

# Type variable for explanation results
E = TypeVar('E', bound='BaseExplanation')


@dataclass
class BaseExplanation(ABC):
    """Base class for all counterfactual explanations."""

    original_sequence: List[int]
    original_prediction: int
    original_prediction_name: str

    @abstractmethod
    def __str__(self) -> str:
        """Return human-readable representation of the explanation."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to dictionary format."""
        pass


@dataclass
class BaseCounterfactual:
    """Base class for individual counterfactual instances."""

    original_sequence: List[int]
    counterfactual_sequence: List[int]
    original_prediction: int
    counterfactual_prediction: int

    def get_changes(self) -> List[Tuple[int, int, int]]:
        """
        Get the changes between original and counterfactual sequences.

        Returns:
            List of (position, original_value, counterfactual_value) tuples
        """
        changes = []
        max_len = max(len(self.original_sequence), len(self.counterfactual_sequence))

        for i in range(max_len):
            orig = self.original_sequence[i] if i < len(self.original_sequence) else None
            cf = self.counterfactual_sequence[i] if i < len(self.counterfactual_sequence) else None
            if orig != cf:
                changes.append((i, orig, cf))

        return changes


class BaseModelPredictor(ABC):
    """
    Abstract base class for model prediction wrappers.

    All counterfactual methods need a consistent interface to query the
    underlying prediction model. This class defines that interface.
    """

    @abstractmethod
    def predict(self, sequence: List[int]) -> int:
        """
        Predict the class index for a sequence.

        Args:
            sequence: Activity sequence (prefix)

        Returns:
            Predicted class index
        """
        pass

    @abstractmethod
    def predict_proba(self, sequence: List[int]) -> np.ndarray:
        """
        Get class probabilities for a sequence.

        Args:
            sequence: Activity sequence (prefix)

        Returns:
            Array of class probabilities
        """
        pass


class BaseCounterfactualMethod(ABC, Generic[E]):
    """
    Abstract base class for counterfactual explanation methods.

    All counterfactual generation algorithms should inherit from this class
    and implement the required methods.
    """

    def __init__(self,
                 model: BaseModelPredictor,
                 vocab_size: int,
                 activity_names: List[str]):
        """
        Args:
            model: Model predictor wrapper
            vocab_size: Number of unique activities
            activity_names: List of activity names
        """
        self.model = model
        self.vocab_size = vocab_size
        self.activity_names = activity_names

    @abstractmethod
    def fit(self, sequences: List[List[int]],
            labels: Optional[List[int]] = None) -> "BaseCounterfactualMethod":
        """
        Fit the method on training data (if required).

        Args:
            sequences: Training sequences
            labels: Optional labels for the sequences

        Returns:
            self
        """
        pass

    @abstractmethod
    def explain(self, sequence: List[int],
                target: Optional[int] = None) -> E:
        """
        Generate counterfactual explanation for a sequence.

        Args:
            sequence: The sequence to explain
            target: Target class (if None, any different class)

        Returns:
            Explanation object
        """
        pass

    def get_activity_name(self, idx: int) -> str:
        """Get activity name for an index."""
        if 0 <= idx < len(self.activity_names):
            return self.activity_names[idx]
        return f"Activity_{idx}"


class ProcessModelPredictor(BaseModelPredictor):
    """
    Model predictor wrapper for U-ED-LSTM and similar process prediction models.

    Provides a consistent interface for querying predictions from the model.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 suffix_step: int = 0,
                 device: Optional[torch.device] = None):
        """
        Args:
            model: The prediction model (e.g., U-ED-LSTM)
            suffix_step: Which suffix step to predict (0 = next activity)
            device: Device to run predictions on
        """
        self.model = model
        self.suffix_step = suffix_step
        self.device = device or next(model.parameters()).device
        self.model.eval()

    def _prepare_input(self, sequence: List[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Convert sequence to model input format.

        This is a basic implementation that only handles the activity sequence.
        Subclass this to add support for additional features.
        """
        activity_tensor = torch.LongTensor([sequence]).to(self.device)
        cat_tensors = [activity_tensor]
        num_tensors = []
        return cat_tensors, num_tensors

    def predict(self, sequence: List[int]) -> int:
        """Predict the next activity class index."""
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().tolist()
        if not sequence:
            return 0

        cat_tensors, num_tensors = self._prepare_input(sequence)

        with torch.no_grad():
            try:
                cat_preds, _ = self.model((cat_tensors, num_tensors))
                activity_logits = cat_preds["Activity_mean"][self.suffix_step]
                return activity_logits.argmax(dim=-1).item()
            except Exception:
                return 0

    def predict_proba(self, sequence: List[int]) -> np.ndarray:
        """Predict class probabilities."""
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().tolist()
        if not sequence:
            return np.array([1.0])

        cat_tensors, num_tensors = self._prepare_input(sequence)

        with torch.no_grad():
            try:
                cat_preds, _ = self.model((cat_tensors, num_tensors))
                activity_logits = cat_preds["Activity_mean"][self.suffix_step]
                probs = torch.softmax(activity_logits, dim=-1)
                return probs.cpu().numpy()[0]
            except Exception:
                return np.array([1.0])

    def predict_batch(self, sequences: List[List[int]]) -> List[int]:
        """Predict for a batch of sequences."""
        return [self.predict(seq) for seq in sequences]


# Evaluation metrics for counterfactuals
def compute_proximity_l1(original: List[int], counterfactual: List[int],
                         vocab_size: int) -> float:
    """Compute L1 distance between sequences (one-hot encoded)."""
    max_len = max(len(original), len(counterfactual))

    # Pad sequences
    orig_padded = original + [0] * (max_len - len(original))
    cf_padded = counterfactual + [0] * (max_len - len(counterfactual))

    # One-hot encode and compute L1
    distance = 0.0
    for o, c in zip(orig_padded, cf_padded):
        if o != c:
            distance += 2.0  # Change from one activity to another = 2 in one-hot

    return distance


def compute_proximity_l2(original: List[int], counterfactual: List[int],
                         vocab_size: int) -> float:
    """Compute L2 distance between sequences (one-hot encoded)."""
    l1 = compute_proximity_l1(original, counterfactual, vocab_size)
    return np.sqrt(l1)


def compute_sparsity_l0(original: List[int], counterfactual: List[int]) -> int:
    """Compute L0 sparsity (number of changed positions)."""
    max_len = max(len(original), len(counterfactual))

    orig_padded = original + [0] * (max_len - len(original))
    cf_padded = counterfactual + [0] * (max_len - len(counterfactual))

    return sum(1 for o, c in zip(orig_padded, cf_padded) if o != c)


def compute_longest_common_prefix(original: List[int],
                                   counterfactual: List[int]) -> int:
    """Compute length of longest common prefix (LCP)."""
    lcp = 0
    for o, c in zip(original, counterfactual):
        if o == c:
            lcp += 1
        else:
            break
    return lcp
