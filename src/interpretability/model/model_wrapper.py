"""
Model wrapper for Integrated Gradients with separate encoder/decoder attribution.

The encoder-decoder model has two distinct input paths:
1. Encoder prefix → encoder LSTM → hidden states (h, c)
2. Decoder SOS → decoder LSTM (using encoder hidden states) → predictions

This wrapper enables computing IG attributions for each path independently,
treating the SOS event as a separate "feature" rather than trying to backprop
through the encoder-decoder boundary.
"""

import torch
from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple, Callable


class IGModelWrapper(nn.Module):
    """
    Wrapper for IG computation with separate encoder and decoder attribution.

    Two attributable inputs:
    - embedded_prefix: Encoder input [seq_len, batch, enc_dim]
    - embedded_sos: Decoder's first input [1, batch, dec_dim]

    Usage:
        wrapper = IGModelWrapper(model, suffix_step=0)

        # Embed inputs
        emb_prefix = wrapper.embed_prefix(prefix)
        emb_sos = wrapper.embed_sos(prefix)

        # Get forward functions for IG
        enc_forward = wrapper.encoder_forward_fn(target_selector, emb_sos)
        sos_forward = wrapper.sos_forward_fn(target_selector, emb_prefix)

        # Compute IG for each
        enc_attrs = ig.compute(emb_prefix, baseline, forward_fn=enc_forward)
        sos_attrs = ig.compute(emb_sos, baseline, forward_fn=sos_forward)
    """

    def __init__(self, model: nn.Module, suffix_step: int = 0):
        super().__init__()
        self.model = model
        self.suffix_step = suffix_step

        # Store embedding references
        self._enc_embeddings = model.embeddings_enc
        self._dec_embeddings = model.embeddings_dec
        self._enc_indices = model.data_indices_enc
        self._dec_indices = model.data_indices_dec

        # Compute feature slices
        self._enc_slices = self._compute_slices(self._enc_embeddings, self._enc_indices)
        self._dec_slices = self._compute_slices(self._dec_embeddings, self._dec_indices)

    def _compute_slices(
        self,
        embeddings: List[nn.Embedding],
        indices: Tuple[List[int], List[int]]
    ) -> Dict[str, Tuple[int, int]]:
        """Compute start/end indices for each feature in concatenated embedding."""
        cat_indices, num_indices = indices
        slices = {'categorical': [], 'numerical': [], 'total_dim': 0}

        offset = 0
        for emb in embeddings:
            slices['categorical'].append((offset, offset + emb.embedding_dim))
            offset += emb.embedding_dim

        for _ in num_indices:
            slices['numerical'].append((offset, offset + 1))
            offset += 1

        slices['total_dim'] = offset
        return slices

    # =========================================================================
    # Embedding methods
    # =========================================================================

    def embed_prefix(self, prefix: Tuple[List[Tensor], List[Tensor]]) -> Tensor:
        """
        Embed prefix sequence using encoder embeddings.

        Args:
            prefix: (cat_tensors, num_tensors) each with shape [batch, seq_len]

        Returns:
            Embedded tensor [seq_len, batch, enc_dim]
        """
        cat_tensors, num_tensors = prefix

        # Select features used by encoder
        cats = [cat_tensors[i] for i in self._enc_indices[0]]
        nums = [num_tensors[i] for i in self._enc_indices[1]]

        # Embed categorical features
        embedded = torch.cat([emb(c) for emb, c in zip(self._enc_embeddings, cats)], dim=-1)

        # Append numerical features
        if nums:
            nums_stacked = torch.cat([n.unsqueeze(-1) for n in nums], dim=-1)
            embedded = torch.cat([embedded, nums_stacked], dim=-1)

        return embedded.permute(1, 0, 2)  # [seq_len, batch, dim]

    def embed_sos(self, prefix: Tuple[List[Tensor], List[Tensor]]) -> Tensor:
        """
        Embed SOS event (last prefix event) using decoder embeddings.

        Args:
            prefix: (cat_tensors, num_tensors) each with shape [batch, seq_len]

        Returns:
            Embedded tensor [1, batch, dec_dim]
        """
        cat_tensors, num_tensors = prefix

        # Take last event from prefix, select features used by decoder
        cats = [cat[:, -1:] for cat in cat_tensors]
        nums = [num[:, -1:] for num in num_tensors]
        cats = [cats[i] for i in self._dec_indices[0]]
        nums = [nums[i] for i in self._dec_indices[1]]

        # Embed categorical features
        embedded = torch.cat([emb(c) for emb, c in zip(self._dec_embeddings, cats)], dim=-1)

        # Append numerical features
        if nums:
            nums_stacked = torch.cat([n.unsqueeze(-1) for n in nums], dim=-1)
            embedded = torch.cat([embedded, nums_stacked], dim=-1)

        return embedded.permute(1, 0, 2)  # [1, batch, dim]

    # =========================================================================
    # Forward pass
    # =========================================================================

    def forward(
        self,
        embedded_prefix: Tensor,
        embedded_sos: Tensor
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Forward pass from embedded inputs.

        Args:
            embedded_prefix: [seq_len, batch, enc_dim]
            embedded_sos: [1, batch, dec_dim]

        Returns:
            (cat_predictions, num_predictions) dicts
        """
        self.model.eval()

        # === Encoder forward ===
        h, c = self._encode(embedded_prefix)

        # === Decoder forward (autoregressive) ===
        return self._decode(embedded_sos, h, c)

    def _encode(self, embedded_prefix: Tensor) -> Tuple[Tensor, Tensor]:
        """Run encoder and return final hidden states."""
        encoder = self.model.encoder

        outputs, (h, c), z = encoder.first_layer(embedded_prefix, hx=None, z=None)
        for layer in encoder.hidden_layers:
            outputs, (h, c), z = layer(outputs, hx=(h, c), z=None)

        return h, c

    def _decode(
        self,
        embedded_sos: Tensor,
        h: Tensor,
        c: Tensor
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Run decoder autoregressively and return predictions."""
        decoder = self.model.decoder
        cat_sizes, num_sizes = decoder.output_sizes

        # Initialize prediction storage
        cat_preds = {f"{k}_{s}": [] for k in cat_sizes for s in ["mean", "var"]}
        num_preds = {f"{k}_{s}": [] for k in num_sizes for s in ["mean", "var"]}

        current_input = embedded_sos
        z = None

        for t in range(self.model.seq_len_pred):
            # Decoder LSTM step
            outputs, (h, c), z = self._decoder_step(current_input, h, c, z)
            final_out = outputs[-1]

            # Compute predictions
            for key in cat_sizes:
                cat_preds[f"{key}_mean"].append(decoder.output_layers[f"{key}_mean"](final_out))
                cat_preds[f"{key}_var"].append(decoder.output_layers[f"{key}_var"](final_out))
            for key in num_sizes:
                num_preds[f"{key}_mean"].append(decoder.output_layers[f"{key}_mean"](final_out))
                num_preds[f"{key}_var"].append(decoder.output_layers[f"{key}_var"](final_out))

            # Prepare next input (soft embedding for gradient flow)
            if t < self.model.seq_len_pred - 1:
                current_input = self._soft_embed_prediction(cat_preds, num_preds, t)

        # Stack predictions: [suffix_len, batch, dim]
        cat_preds = {k: torch.stack(v, dim=0) for k, v in cat_preds.items()}
        num_preds = {k: torch.stack(v, dim=0) for k, v in num_preds.items()}

        return cat_preds, num_preds

    def _decoder_step(
        self,
        embedded_input: Tensor,
        h: Tensor,
        c: Tensor,
        z: Optional[Tuple]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tuple]:
        """Single decoder LSTM step."""
        decoder = self.model.decoder

        if z is None:
            outputs, (h, c), z_first = decoder.first_layer(embedded_input, hx=(h, c), z=None)
            z_hidden = []
            for layer in decoder.hidden_layers:
                outputs, (h, c), z_h = layer(outputs, hx=(h, c), z=None)
                z_hidden.append(z_h)
            z = (z_first, z_hidden)
        else:
            outputs, (h, c), _ = decoder.first_layer(embedded_input, hx=(h, c), z=z[0])
            for i, layer in enumerate(decoder.hidden_layers):
                outputs, (h, c), _ = layer(outputs, hx=(h, c), z=z[1][i])

        return outputs, (h, c), z

    def _soft_embed_prediction(
        self,
        cat_preds: Dict[str, List[Tensor]],
        num_preds: Dict[str, List[Tensor]],
        step: int
    ) -> Tensor:
        """
        Create soft embedding from prediction for gradient flow.
        Uses weighted sum of embeddings by softmax probabilities.
        """
        cat_sizes, num_sizes = self.model.decoder.output_sizes
        parts = []

        # Soft-embed categorical predictions
        for i, key in enumerate(cat_sizes):
            logits = cat_preds[f"{key}_mean"][step]
            probs = torch.softmax(logits, dim=-1)
            weights = self._dec_embeddings[i].weight
            soft_emb = torch.matmul(probs, weights)
            parts.append(soft_emb.unsqueeze(0))

        # Numerical predictions directly
        for key in num_sizes:
            parts.append(num_preds[f"{key}_mean"][step].unsqueeze(0))

        return torch.cat(parts, dim=-1)

    # =========================================================================
    # Forward functions for IG
    # =========================================================================

    def encoder_forward_fn(
        self,
        target_selector: 'TargetSelector',
        embedded_sos: Tensor
    ) -> Callable[[Tensor], Tensor]:
        """
        Create forward function for IG attribution w.r.t. encoder prefix.

        The SOS embedding is fixed; gradients flow only through the prefix.
        """
        fixed_sos = embedded_sos.detach()

        def forward(embedded_prefix: Tensor) -> Tensor:
            preds = self.forward(embedded_prefix, fixed_sos)
            return target_selector.select(preds, self.suffix_step)

        return forward

    def sos_forward_fn(
        self,
        target_selector: 'TargetSelector',
        embedded_prefix: Tensor
    ) -> Callable[[Tensor], Tensor]:
        """
        Create forward function for IG attribution w.r.t. SOS event.

        The prefix embedding is fixed; gradients flow only through SOS.
        """
        fixed_prefix = embedded_prefix.detach()

        def forward(embedded_sos: Tensor) -> Tensor:
            preds = self.forward(fixed_prefix, embedded_sos)
            return target_selector.select(preds, self.suffix_step)

        return forward

    def combined_forward_fn(
        self,
        target_selector: 'TargetSelector'
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """
        Create forward function for joint IG (both inputs attributable).
        """
        def forward(embedded_prefix: Tensor, embedded_sos: Tensor) -> Tensor:
            preds = self.forward(embedded_prefix, embedded_sos)
            return target_selector.select(preds, self.suffix_step)

        return forward

    # =========================================================================
    # Decoder chain attribution (autoregressive)
    # =========================================================================

    def get_decoder_inputs_for_step(
        self,
        embedded_prefix: Tensor,
        embedded_sos: Tensor,
        target_step: int
    ) -> Tensor:
        """
        Get all decoder inputs needed for predicting suffix step `target_step`.

        For step 0: returns [SOS]
        For step 1: returns [SOS, soft_embed(pred_0)]
        For step t: returns [SOS, soft_embed(pred_0), ..., soft_embed(pred_{t-1})]

        Returns:
            Tensor of shape [target_step + 1, batch, dec_dim]
        """
        self.model.eval()
        with torch.no_grad():
            h, c = self._encode(embedded_prefix.detach())

            decoder_inputs = [embedded_sos.detach()]
            current_input = embedded_sos.detach()
            z = None

            decoder = self.model.decoder
            cat_sizes, num_sizes = decoder.output_sizes

            for t in range(target_step):
                # Run decoder step
                outputs, (h, c), z = self._decoder_step(current_input, h, c, z)
                final_out = outputs[-1]

                # Compute predictions for this step
                cat_preds_t = {}
                num_preds_t = {}
                for key in cat_sizes:
                    cat_preds_t[f"{key}_mean"] = [decoder.output_layers[f"{key}_mean"](final_out)]
                for key in num_sizes:
                    num_preds_t[f"{key}_mean"] = [decoder.output_layers[f"{key}_mean"](final_out)]

                # Create soft embedding for next input
                next_input = self._soft_embed_prediction(cat_preds_t, num_preds_t, 0)
                decoder_inputs.append(next_input.detach())
                current_input = next_input.detach()

            return torch.cat(decoder_inputs, dim=0)  # [target_step + 1, batch, dec_dim]

    def forward_from_decoder_sequence(
        self,
        embedded_prefix: Tensor,
        decoder_inputs: Tensor,
        target_step: int
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Forward pass using pre-specified decoder inputs.

        Args:
            embedded_prefix: [seq_len, batch, enc_dim]
            decoder_inputs: [num_steps, batch, dec_dim] - all decoder inputs
            target_step: which suffix step to compute

        Returns:
            (cat_predictions, num_predictions) for the target step
        """
        self.model.eval()
        h, c = self._encode(embedded_prefix)

        decoder = self.model.decoder
        cat_sizes, num_sizes = decoder.output_sizes
        z = None

        # Process each decoder input up to target_step
        for t in range(target_step + 1):
            current_input = decoder_inputs[t:t+1]  # [1, batch, dec_dim]
            outputs, (h, c), z = self._decoder_step(current_input, h, c, z)

        # Get final output and compute predictions
        final_out = outputs[-1]
        cat_preds = {}
        num_preds = {}
        for key in cat_sizes:
            cat_preds[f"{key}_mean"] = decoder.output_layers[f"{key}_mean"](final_out)
            cat_preds[f"{key}_var"] = decoder.output_layers[f"{key}_var"](final_out)
        for key in num_sizes:
            num_preds[f"{key}_mean"] = decoder.output_layers[f"{key}_mean"](final_out)
            num_preds[f"{key}_var"] = decoder.output_layers[f"{key}_var"](final_out)

        return cat_preds, num_preds

    def decoder_chain_forward_fn(
        self,
        target_selector: 'TargetSelector',
        embedded_prefix: Tensor
    ) -> Callable[[Tensor], Tensor]:
        """
        Create forward function for IG attribution w.r.t. decoder input sequence.

        The prefix is fixed; gradients flow through the decoder inputs.
        """
        fixed_prefix = embedded_prefix.detach()

        def forward(decoder_inputs: Tensor) -> Tensor:
            preds = self.forward_from_decoder_sequence(
                fixed_prefix, decoder_inputs, self.suffix_step
            )
            return target_selector.select_direct(preds)

        return forward

    # =========================================================================
    # Feature information
    # =========================================================================

    def get_encoder_feature_info(self) -> Dict[str, List[Tuple[str, int, int]]]:
        """
        Get encoder feature names and their slice positions.

        Returns:
            Dict with 'categorical' and 'numerical' keys,
            each containing list of (name, start_idx, end_idx) tuples.
        """
        enc_cat_feats = self.model.enc_feat[0]
        enc_num_feats = self.model.enc_feat[1]

        return {
            'categorical': [
                (name, *self._enc_slices['categorical'][i])
                for i, name in enumerate(enc_cat_feats)
            ],
            'numerical': [
                (name, *self._enc_slices['numerical'][i])
                for i, name in enumerate(enc_num_feats)
            ]
        }

    def get_decoder_feature_info(self) -> Dict[str, List[Tuple[str, int, int]]]:
        """
        Get decoder feature names and their slice positions.

        Returns:
            Dict with 'categorical' and 'numerical' keys,
            each containing list of (name, start_idx, end_idx) tuples.
        """
        dec_cat_feats = self.model.dec_feat[0]
        dec_num_feats = self.model.dec_feat[1]

        return {
            'categorical': [
                (name, *self._dec_slices['categorical'][i])
                for i, name in enumerate(dec_cat_feats)
            ],
            'numerical': [
                (name, *self._dec_slices['numerical'][i])
                for i, name in enumerate(dec_num_feats)
            ]
        }

    # Backwards compatibility aliases
    def get_feature_names(self) -> Dict[str, List[Tuple[str, int, int]]]:
        """Alias for get_encoder_feature_info()."""
        return self.get_encoder_feature_info()

    def get_decoder_feature_names(self) -> Dict[str, List[Tuple[str, int, int]]]:
        """Alias for get_decoder_feature_info()."""
        return self.get_decoder_feature_info()

    # Legacy method for backwards compatibility
    def embed_event(
        self,
        event: Tuple[List[Tensor], List[Tensor]],
        is_prediction: bool = False
    ) -> Tensor:
        """
        Legacy method: Embed a single event using decoder embeddings.
        Prefer using embed_sos() for the SOS event.
        """
        cat_tensors, num_tensors = event

        if is_prediction:
            cats = cat_tensors
            nums = num_tensors
        else:
            cats = [cat_tensors[i] for i in self._dec_indices[0]]
            nums = [num_tensors[i] for i in self._dec_indices[1]]

        embedded = torch.cat([emb(c) for emb, c in zip(self._dec_embeddings, cats)], dim=-1)

        if nums:
            nums_stacked = torch.cat([n.unsqueeze(-1) for n in nums], dim=-1)
            embedded = torch.cat([embedded, nums_stacked], dim=-1)

        return embedded.permute(1, 0, 2)

    # Legacy method for backwards compatibility
    def forward_from_embeddings(
        self,
        embedded_prefix: Tensor,
        embedded_sos: Optional[Tensor] = None
    ) -> List[Dict[str, Tensor]]:
        """
        Legacy method: Forward pass from embeddings.
        Returns [cat_predictions, num_predictions] for backwards compatibility.
        """
        if embedded_sos is None:
            embedded_sos = embedded_prefix[-1:, :, :]

        cat_preds, num_preds = self.forward(embedded_prefix, embedded_sos)
        return [cat_preds, num_preds]

    # Legacy method for backwards compatibility
    def create_forward_fn(
        self,
        target_selector: 'TargetSelector',
        prefix: Tuple[List[Tensor], List[Tensor]]
    ) -> Callable[[Tensor], Tensor]:
        """Legacy method: Create forward function for IG w.r.t. prefix."""
        embedded_sos = self.embed_sos(prefix)
        return self.encoder_forward_fn(target_selector, embedded_sos)
