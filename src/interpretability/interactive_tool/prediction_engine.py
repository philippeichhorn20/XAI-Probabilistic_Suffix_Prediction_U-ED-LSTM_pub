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
    """Handles model inference for suffix prediction."""

    def __init__(
        self,
        model,
        cat_feature_names: List[str],
        num_feature_names: List[str],
        activity_feature: str,
        idx_to_activity: Dict[int, str],
        activity_to_idx: Dict[str, int],
        device: str = 'cpu'
    ):
        self.model = model
        self.cat_feature_names = cat_feature_names
        self.num_feature_names = num_feature_names
        self.activity_feature = activity_feature
        self.idx_to_activity = idx_to_activity
        self.activity_to_idx = activity_to_idx
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # Get EOS token index
        self.eos_idx = activity_to_idx.get('EOS', activity_to_idx.get('<EOS>', None))

        # Create case editor
        self.case_editor = CaseEditor(cat_feature_names, num_feature_names)

        # Get decoder feature names from model
        self.dec_cat_names = list(model.dec_feat[0])
        self.dec_num_names = list(model.dec_feat[1])

    def predict_suffix(
        self,
        prefix: Tuple[List[torch.Tensor], List[torch.Tensor]],
        max_suffix_length: int = 20,
        top_k: int = 5,
        actual_suffix: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None
    ) -> PredictionResult:
        """
        Predict suffix given a prefix.

        Args:
            prefix: Tuple of (categorical_tensors, numerical_tensors)
            max_suffix_length: Maximum length of predicted suffix
            top_k: Number of top predictions to return per step
            actual_suffix: Ground truth suffix if available

        Returns:
            PredictionResult with predictions and probabilities
        """
        self.model.eval()

        with torch.no_grad():
            # Prepare prefix tensors - add batch dimension
            cat_tensors = [t.unsqueeze(0).to(self.device) for t in prefix[0]]
            num_tensors = [t.unsqueeze(0).to(self.device) for t in prefix[1]]

            # Get encoder output - encoder expects [cat_tensors, num_tensors] as single list
            encoder_output = self.model.encoder([cat_tensors, num_tensors])
            # encoder_output is (h, c) tuple

            # Initialize decoder input with last event from prefix
            batch_size = 1

            # Create initial decoder input from last prefix event
            dec_cat_inputs = []
            for name in self.dec_cat_names:
                # Find this feature in the prefix
                if name in self.cat_feature_names:
                    feat_idx = self.cat_feature_names.index(name)
                    if feat_idx < len(cat_tensors):
                        # Use last value from prefix
                        last_val = cat_tensors[feat_idx][:, -1:].clone()  # [batch, 1]
                    else:
                        last_val = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
                else:
                    last_val = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
                dec_cat_inputs.append(last_val)

            dec_num_inputs = []
            for name in self.dec_num_names:
                if name in self.num_feature_names:
                    feat_idx = self.num_feature_names.index(name)
                    if feat_idx < len(num_tensors):
                        last_val = num_tensors[feat_idx][:, -1:].clone()  # [batch, 1]
                    else:
                        last_val = torch.zeros(batch_size, 1, device=self.device)
                else:
                    last_val = torch.zeros(batch_size, 1, device=self.device)
                dec_num_inputs.append(last_val)

            # Autoregressive decoding
            predicted_indices = []
            all_probabilities = []
            all_top_k = []

            hx = encoder_output  # (h, c) from encoder
            z = None  # Dropout mask, None for first call

            for step in range(max_suffix_length):
                # Decoder forward pass
                # decoder expects: input=(cats, nums), hx=(h,c), z=None, pred=True
                predictions, hx, z = self.model.decoder(
                    input=(dec_cat_inputs, dec_num_inputs),
                    hx=hx,
                    z=z,
                    pred=True
                )

                # predictions = [[cat_means, num_means], [cat_vars, num_vars]]
                # cat_means is a dict like {"Activity_mean": logits_tensor, "Resource_mean": ...}
                cat_means = predictions[0][0]
                num_means = predictions[0][1]

                # Get activity logits
                activity_key = f"{self.activity_feature}_mean"
                if activity_key not in cat_means:
                    # Try without _mean suffix
                    activity_key = self.activity_feature + "_mean"

                activity_logits = cat_means[activity_key]  # [batch, num_activities]
                activity_logits = activity_logits.squeeze(0)  # [num_activities]

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

                # Prepare next decoder input using predictions
                dec_cat_inputs = []
                for name in self.dec_cat_names:
                    key = f"{name}_mean"
                    if key in cat_means:
                        logits = cat_means[key].squeeze(0)  # [num_classes]
                        pred_val = logits.argmax().item()
                        next_input = torch.tensor([[pred_val]], dtype=torch.long, device=self.device)
                    else:
                        next_input = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
                    dec_cat_inputs.append(next_input)

                dec_num_inputs = []
                for name in self.dec_num_names:
                    key = f"{name}_mean"
                    if key in num_means:
                        pred_val = num_means[key].squeeze(0)  # scalar or [1]
                        if pred_val.dim() == 0:
                            pred_val = pred_val.unsqueeze(0)
                        next_input = pred_val.unsqueeze(0)  # [batch, 1]
                    else:
                        next_input = torch.zeros(batch_size, 1, device=self.device)
                    dec_num_inputs.append(next_input)

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
