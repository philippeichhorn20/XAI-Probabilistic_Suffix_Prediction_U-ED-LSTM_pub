"""Attribution engine for computing integrated gradients."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class AttributionResult:
    """Result of attribution computation."""
    # Per-event, per-feature attribution scores
    event_attributions: List[Dict[str, float]]  # List of dicts, one per event
    feature_names: List[str]
    prefix_length: int
    target_activity: str
    target_activity_idx: int

    def to_heatmap_data(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Convert to numpy array for heatmap visualization."""
        if not self.event_attributions:
            return np.array([[]]), [], []

        features = list(self.event_attributions[0].keys())
        data = np.array([
            [event.get(f, 0.0) for f in features]
            for event in self.event_attributions
        ])
        event_labels = [f"Event {i+1}" for i in range(len(self.event_attributions))]
        return data, event_labels, features


class AttributionEngine:
    """Computes attributions using integrated gradients."""

    def __init__(
        self,
        model,
        cat_feature_names: List[str],
        num_feature_names: List[str],
        activity_feature: str,
        idx_to_activity: Dict[int, str],
        device: str = 'cpu'
    ):
        self.model = model
        self.cat_feature_names = cat_feature_names
        self.num_feature_names = num_feature_names
        self.activity_feature = activity_feature
        self.idx_to_activity = idx_to_activity
        self.device = torch.device(device)

        # Get encoder feature names
        self.enc_cat_names = list(model.enc_feat[0])
        self.enc_num_names = list(model.enc_feat[1])

    def compute_simple_gradient_attribution(
        self,
        prefix_cat: List[torch.Tensor],
        prefix_num: List[torch.Tensor],
        target_activity_idx: Optional[int] = None
    ) -> AttributionResult:
        """
        Compute simple gradient-based attribution.
        This is a faster alternative to full integrated gradients.
        """
        self.model.eval()

        # Prepare tensors with gradients
        cat_tensors = [t.unsqueeze(0).clone().detach().to(self.device) for t in prefix_cat]
        num_tensors = [t.unsqueeze(0).clone().detach().float().to(self.device) for t in prefix_num]

        # Enable gradients for numerical features
        for t in num_tensors:
            t.requires_grad_(True)

        # Get encoder embeddings and enable gradients
        embedded_cats = []
        for i, (tensor, emb) in enumerate(zip(cat_tensors, self.model.embeddings_enc)):
            if i < len(self.model.embeddings_enc):
                embedded = emb(tensor)
                embedded.requires_grad_(True)
                embedded_cats.append(embedded)

        # Concatenate embeddings
        if embedded_cats:
            cat_embedded = torch.cat(embedded_cats, dim=-1)  # [batch, seq, embed_dim]
        else:
            cat_embedded = torch.zeros(1, cat_tensors[0].shape[1], 0, device=self.device)

        if num_tensors:
            num_stacked = torch.stack([t.squeeze(-1) if t.dim() > 2 else t for t in num_tensors], dim=-1)
            if num_stacked.dim() == 2:
                num_stacked = num_stacked.unsqueeze(0)
        else:
            num_stacked = torch.zeros(1, cat_tensors[0].shape[1], 0, device=self.device)

        # Combine for encoder input
        encoder_input = torch.cat([cat_embedded, num_stacked], dim=-1)  # [batch, seq, features]
        encoder_input = encoder_input.permute(1, 0, 2)  # [seq, batch, features]

        # Forward through encoder layers manually
        outputs, (h, c), _ = self.model.encoder.first_layer(input=encoder_input, hx=None, z=None)
        for layer in self.model.encoder.hidden_layers:
            outputs, (h, c), _ = layer(input=outputs, hx=(h, c), z=None)

        # Decoder forward for first prediction
        dec_cat_inputs = []
        for name in self.model.dec_feat[0]:
            if name in self.cat_feature_names:
                feat_idx = self.cat_feature_names.index(name)
                if feat_idx < len(cat_tensors):
                    dec_cat_inputs.append(cat_tensors[feat_idx][:, -1:])
                else:
                    dec_cat_inputs.append(torch.zeros(1, 1, dtype=torch.long, device=self.device))
            else:
                dec_cat_inputs.append(torch.zeros(1, 1, dtype=torch.long, device=self.device))

        dec_num_inputs = []
        for name in self.model.dec_feat[1]:
            if name in self.num_feature_names:
                feat_idx = self.num_feature_names.index(name)
                if feat_idx < len(num_tensors):
                    dec_num_inputs.append(num_tensors[feat_idx][:, -1:])
                else:
                    dec_num_inputs.append(torch.zeros(1, 1, device=self.device))
            else:
                dec_num_inputs.append(torch.zeros(1, 1, device=self.device))

        predictions, _, _ = self.model.decoder(
            input=(dec_cat_inputs, dec_num_inputs),
            hx=(h, c),
            z=None,
            pred=True
        )

        # Get activity logits
        activity_logits = predictions[0][0][f"{self.activity_feature}_mean"]

        # Determine target
        if target_activity_idx is None:
            target_activity_idx = activity_logits.argmax().item()

        # Get target logit
        target_logit = activity_logits[0, target_activity_idx]

        # Compute gradients
        target_logit.backward(retain_graph=True)

        # Extract attributions per event
        seq_len = cat_tensors[0].shape[1]
        event_attributions = []

        for t in range(seq_len):
            attr_dict = {}

            # Categorical feature attributions (from embeddings)
            embed_idx = 0
            for i, name in enumerate(self.enc_cat_names):
                if i < len(embedded_cats) and embedded_cats[i].grad is not None:
                    grad = embedded_cats[i].grad[0, t, :].abs().mean().item()
                    attr_dict[name] = grad
                else:
                    attr_dict[name] = 0.0

            # Numerical feature attributions
            for i, name in enumerate(self.enc_num_names):
                if i < len(num_tensors) and num_tensors[i].grad is not None:
                    grad = num_tensors[i].grad[0, t].abs().item() if num_tensors[i].grad.dim() > 1 else num_tensors[i].grad[0].abs().item()
                    attr_dict[name] = grad
                else:
                    attr_dict[name] = 0.0

            event_attributions.append(attr_dict)

        return AttributionResult(
            event_attributions=event_attributions,
            feature_names=self.enc_cat_names + self.enc_num_names,
            prefix_length=seq_len,
            target_activity=self.idx_to_activity.get(target_activity_idx, f"Unknown_{target_activity_idx}"),
            target_activity_idx=target_activity_idx
        )

    def compute_input_perturbation_attribution(
        self,
        prefix_cat: List[torch.Tensor],
        prefix_num: List[torch.Tensor],
        target_activity_idx: Optional[int] = None
    ) -> AttributionResult:
        """
        Compute attribution using input perturbation (leave-one-out).
        More interpretable but slower than gradient methods.
        """
        self.model.eval()

        with torch.no_grad():
            # Get baseline prediction
            cat_tensors = [t.unsqueeze(0).to(self.device) for t in prefix_cat]
            num_tensors = [t.unsqueeze(0).to(self.device) for t in prefix_num]

            encoder_output = self.model.encoder([cat_tensors, num_tensors])

            # Decoder forward
            dec_cat_inputs = []
            for name in self.model.dec_feat[0]:
                if name in self.cat_feature_names:
                    feat_idx = self.cat_feature_names.index(name)
                    if feat_idx < len(cat_tensors):
                        dec_cat_inputs.append(cat_tensors[feat_idx][:, -1:])
                    else:
                        dec_cat_inputs.append(torch.zeros(1, 1, dtype=torch.long, device=self.device))
                else:
                    dec_cat_inputs.append(torch.zeros(1, 1, dtype=torch.long, device=self.device))

            dec_num_inputs = []
            for name in self.model.dec_feat[1]:
                if name in self.num_feature_names:
                    feat_idx = self.num_feature_names.index(name)
                    if feat_idx < len(num_tensors):
                        dec_num_inputs.append(num_tensors[feat_idx][:, -1:])
                    else:
                        dec_num_inputs.append(torch.zeros(1, 1, device=self.device))
                else:
                    dec_num_inputs.append(torch.zeros(1, 1, device=self.device))

            predictions, _, _ = self.model.decoder(
                input=(dec_cat_inputs, dec_num_inputs),
                hx=encoder_output,
                z=None,
                pred=True
            )

            activity_logits = predictions[0][0][f"{self.activity_feature}_mean"]

            if target_activity_idx is None:
                target_activity_idx = activity_logits.argmax().item()

            baseline_prob = torch.softmax(activity_logits, dim=-1)[0, target_activity_idx].item()

            # Compute leave-one-out attributions
            seq_len = prefix_cat[0].shape[0]
            event_attributions = []

            for t in range(seq_len):
                attr_dict = {}

                # For each feature, zero out at timestep t and measure change
                for feat_idx, name in enumerate(self.enc_cat_names):
                    if feat_idx < len(prefix_cat):
                        # Create perturbed input
                        perturbed_cat = [tensor.clone() for tensor in prefix_cat]
                        perturbed_cat[feat_idx][t] = 0  # Zero out this feature at time t

                        perturbed_cat_tensors = [pt.unsqueeze(0).to(self.device) for pt in perturbed_cat]

                        # Forward pass
                        enc_out = self.model.encoder([perturbed_cat_tensors, num_tensors])
                        preds, _, _ = self.model.decoder(
                            input=(dec_cat_inputs, dec_num_inputs),
                            hx=enc_out,
                            z=None,
                            pred=True
                        )

                        perturbed_logits = preds[0][0][f"{self.activity_feature}_mean"]
                        perturbed_prob = torch.softmax(perturbed_logits, dim=-1)[0, target_activity_idx].item()

                        # Attribution = change in probability
                        attr_dict[name] = abs(baseline_prob - perturbed_prob)
                    else:
                        attr_dict[name] = 0.0

                # Numerical features
                for feat_idx, name in enumerate(self.enc_num_names):
                    if feat_idx < len(prefix_num):
                        perturbed_num = [tensor.clone() for tensor in prefix_num]
                        perturbed_num[feat_idx][t] = 0.0

                        perturbed_num_tensors = [pt.unsqueeze(0).to(self.device) for pt in perturbed_num]

                        enc_out = self.model.encoder([cat_tensors, perturbed_num_tensors])
                        preds, _, _ = self.model.decoder(
                            input=(dec_cat_inputs, dec_num_inputs),
                            hx=enc_out,
                            z=None,
                            pred=True
                        )

                        perturbed_logits = preds[0][0][f"{self.activity_feature}_mean"]
                        perturbed_prob = torch.softmax(perturbed_logits, dim=-1)[0, target_activity_idx].item()

                        attr_dict[name] = abs(baseline_prob - perturbed_prob)
                    else:
                        attr_dict[name] = 0.0

                event_attributions.append(attr_dict)

        return AttributionResult(
            event_attributions=event_attributions,
            feature_names=self.enc_cat_names + self.enc_num_names,
            prefix_length=seq_len,
            target_activity=self.idx_to_activity.get(target_activity_idx, f"Unknown_{target_activity_idx}"),
            target_activity_idx=target_activity_idx
        )
