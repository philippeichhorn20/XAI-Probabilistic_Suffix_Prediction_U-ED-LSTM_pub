"""Prediction engine for running model inference."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from .case_editor import EditableCase, CaseEditor


@dataclass
class PredictionResult:
    """Result of a suffix prediction."""
    predicted_activities: List[str]  # Human-readable activity names
    predicted_activity_indices: List[int]  # Encoded indices
    activity_probabilities: List[Dict[str, float]]  # Per-step probability distributions
    top_k_predictions: List[List[Tuple[str, float]]]  # Top-k activities per step
    actual_suffix_activities: Optional[List[str]] = None  # Ground truth if available

    @property
    def predicted_sequence_str(self) -> str:
        """Get predicted sequence as arrow-separated string."""
        return " → ".join(self.predicted_activities)

    @property
    def actual_sequence_str(self) -> Optional[str]:
        """Get actual sequence as arrow-separated string."""
        if self.actual_suffix_activities:
            return " → ".join(self.actual_suffix_activities)
        return None


class PredictionEngine:
    """Handles model inference for suffix prediction.

    Uses model.inference() to match the evaluation pipeline exactly:
    - Prefix tensors are left-padded with zeros to window_size
    - First decoder call uses pred=False (SOS event from prefix)
    - Subsequent decoder calls use pred=True (predicted events)
    """

    def __init__(
        self,
        model,
        cat_feature_names: List[str],
        num_feature_names: List[str],
        activity_feature: str,
        idx_to_activity: Dict[int, str],
        activity_to_idx: Dict[str, int],
        window_size: int,
        device: str = 'cpu'
    ):
        self.model = model
        self.cat_feature_names = cat_feature_names
        self.num_feature_names = num_feature_names
        self.activity_feature = activity_feature
        self.idx_to_activity = idx_to_activity
        self.activity_to_idx = activity_to_idx
        self.window_size = window_size
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # Get EOS token index
        self.eos_idx = activity_to_idx.get('EOS', activity_to_idx.get('<EOS>', None))

        # Create case editor
        self.case_editor = CaseEditor(cat_feature_names, num_feature_names)

    def _left_pad_tensors(
        self,
        tensors: List[torch.Tensor],
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        """Left-pad a list of 1D tensors to window_size with zeros, adding batch dim.

        Input tensors have shape (seq_len,).
        Output tensors have shape (1, window_size).
        """
        padded = []
        for t in tensors:
            p = torch.zeros(1, self.window_size, dtype=dtype, device=self.device)
            seq_len = t.shape[0]
            p[0, -seq_len:] = t.to(self.device)
            padded.append(p)
        return padded

    def predict_suffix(
        self,
        prefix: Tuple[List[torch.Tensor], List[torch.Tensor]],
        max_suffix_length: int = 20,
        top_k: int = 5,
        actual_suffix: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None
    ) -> PredictionResult:
        """
        Predict suffix given a prefix.

        Uses model.inference() which correctly handles the encoder-decoder
        handoff (SOS event with pred=False for the first step).

        Args:
            prefix: Tuple of (categorical_tensors, numerical_tensors),
                    each tensor has shape (prefix_length,) without batch dim.
            max_suffix_length: Maximum length of predicted suffix
            top_k: Number of top predictions to return per step
            actual_suffix: Ground truth suffix if available

        Returns:
            PredictionResult with predictions and probabilities
        """
        self.model.eval()

        with torch.no_grad():
            # Left-pad prefix tensors to window_size (matching training format)
            padded_cat = self._left_pad_tensors(prefix[0], dtype=torch.long)
            padded_num = self._left_pad_tensors(prefix[1], dtype=torch.float32)
            padded_prefix = [padded_cat, padded_num]

            # First prediction: model.inference handles encoder + SOS + first decoder call
            prediction, (h, c), z = self.model.inference(prefix=padded_prefix)

            predicted_indices = []
            all_probabilities = []
            all_top_k = []

            for step in range(max_suffix_length):
                # prediction[0][0] = dict of categorical logits {name_mean: tensor}
                # prediction[0][1] = dict of numerical predictions {name_mean: tensor}
                cat_means = prediction[0][0]
                num_means = prediction[0][1]

                # Get activity logits
                activity_key = f"{self.activity_feature}_mean"
                activity_logits = cat_means[activity_key].squeeze(0)  # [num_classes]
                probs = F.softmax(activity_logits, dim=-1)

                # Get predicted activity
                pred_idx = probs.argmax().item()
                predicted_indices.append(pred_idx)

                # Store probability distribution
                prob_dict = {
                    self.idx_to_activity.get(i, f"Unknown_{i}"): probs[i].item()
                    for i in range(len(probs))
                }
                all_probabilities.append(prob_dict)

                # Get top-k predictions
                top_k_vals, top_k_indices = probs.topk(min(top_k, len(probs)))
                top_k_list = [
                    (self.idx_to_activity.get(idx.item(), f"Unknown_{idx.item()}"), val.item())
                    for idx, val in zip(top_k_indices, top_k_vals)
                ]
                all_top_k.append(top_k_list)

                # Check for EOS
                if self.eos_idx is not None and pred_idx == self.eos_idx:
                    break

                # Create last_event from prediction for next decoder step
                # (matches evaluation.py's pattern exactly)
                cat_prediction = {
                    k: torch.argmax(v, keepdim=True)
                    for k, v in cat_means.items()
                }
                num_prediction = {k: v for k, v in num_means.items()}

                last_event = (
                    list(cat_prediction.values()),
                    list(num_prediction.values()),
                )

                # Subsequent predictions use model.inference with last_event
                prediction, (h, c) = self.model.inference(
                    last_event=last_event,
                    hx=(h, c),
                    z=z,
                )

        # Convert indices to activity names
        predicted_activities = [
            self.idx_to_activity.get(idx, f"Unknown_{idx}")
            for idx in predicted_indices
        ]

        # Get actual suffix activities if provided
        actual_activities = None
        if actual_suffix is not None:
            actual_cat, _ = actual_suffix
            if len(actual_cat) > 0 and len(actual_cat[0]) > 0:
                activity_idx = self.cat_feature_names.index(self.activity_feature)
                if activity_idx < len(actual_cat):
                    actual_activities = [
                        self.idx_to_activity.get(idx.item(), f"Unknown_{idx.item()}")
                        for idx in actual_cat[activity_idx]
                    ]

        return PredictionResult(
            predicted_activities=predicted_activities,
            predicted_activity_indices=predicted_indices,
            activity_probabilities=all_probabilities,
            top_k_predictions=all_top_k,
            actual_suffix_activities=actual_activities
        )

    def predict_from_editable_case(
        self,
        case: EditableCase,
        prefix_length: int,
        max_suffix_length: int = 20,
        top_k: int = 5
    ) -> PredictionResult:
        """
        Predict suffix from an editable case.

        Args:
            case: EditableCase to predict from
            prefix_length: Number of events to use as prefix
            max_suffix_length: Maximum suffix length
            top_k: Number of top predictions per step

        Returns:
            PredictionResult
        """
        prefix_tensors, suffix_tensors = self.case_editor.create_prefix_suffix_split(
            case, prefix_length
        )

        return self.predict_suffix(
            prefix=prefix_tensors,
            max_suffix_length=max_suffix_length,
            top_k=top_k,
            actual_suffix=suffix_tensors if len(suffix_tensors[0]) > 0 and len(suffix_tensors[0][0]) > 0 else None
        )
