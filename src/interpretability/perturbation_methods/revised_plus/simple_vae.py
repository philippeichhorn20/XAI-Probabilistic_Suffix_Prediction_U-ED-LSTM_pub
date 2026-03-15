"""
LSTM-VAE for process sequences.

Architecture aligned with Stevens et al. (2024) REVISED+:
- Bidirectional LSTM encoder (richer sequence representations)
- Autoregressive LSTM decoder with teacher forcing
- CE loss for categoricals (single-label per position)
- Dropout, weight decay, conservative gradient clipping

Data is right-padded (events at start, padding at end) for consistent
autoregressive decode between training and inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

# Import existing constraint infrastructure
from ..declare import DeclareConstraint, DeclareConstraintChecker


def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Create sinusoidal positional encoding matrix (max_len, d_model)."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model > 1:
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
    return pe


class SimpleSequenceVAE(nn.Module):
    """Bidirectional LSTM-VAE with autoregressive decoder for process sequences."""

    def __init__(
        self,
        cat_vocab_sizes: List[int],
        num_features: int,
        seq_len: int,
        embed_dim: int = 24,
        encoder_hidden: int = 100,
        decoder_hidden: int = 50,
        latent_dim: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        pos_dim: int = 16,
        # Legacy compat: accept hidden_dim and map to encoder_hidden
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        # Legacy support: if hidden_dim provided but not encoder_hidden, use it
        if hidden_dim is not None and encoder_hidden == 100:
            encoder_hidden = hidden_dim
            decoder_hidden = hidden_dim

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.num_layers = num_layers
        self.cat_vocab_sizes = cat_vocab_sizes
        self.num_features = num_features
        self.pos_dim = pos_dim
        self.embed_dim = embed_dim
        # Keep for backward compat with code that reads vae.hidden_dim
        self.hidden_dim = encoder_hidden

        # Embeddings for categorical features (padding_idx=0)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for vocab_size in cat_vocab_sizes
        ])

        self.input_dim = len(cat_vocab_sizes) * embed_dim + num_features

        # Bidirectional LSTM encoder
        self.encoder = nn.LSTM(
            self.input_dim, encoder_hidden,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        # BiLSTM outputs 2 * encoder_hidden; project to latent
        self.fc_mu = nn.Linear(2 * encoder_hidden, latent_dim)
        self.fc_logvar = nn.Linear(2 * encoder_hidden, latent_dim)

        # Sinusoidal positional encoding (frozen buffer)
        self.register_buffer('pos_encoding', _sinusoidal_encoding(seq_len, pos_dim))

        # Decoder embedding for autoregressive input (activity feature only)
        self.decoder_embedding = nn.Embedding(cat_vocab_sizes[0], embed_dim, padding_idx=0)

        # Autoregressive decoder: z + pos + prev_activity_emb → LSTM
        self.decoder = nn.LSTM(
            latent_dim + pos_dim + embed_dim, decoder_hidden,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dec_dropout = nn.Dropout(dropout)

        # Output heads (from decoder_hidden, not encoder_hidden)
        self.cat_heads = nn.ModuleList([
            nn.Linear(decoder_hidden, vocab_size) for vocab_size in cat_vocab_sizes
        ])
        self.num_head = nn.Linear(decoder_hidden, num_features) if num_features > 0 else None

    def _embed_inputs(self, cat_inputs: List[torch.Tensor], num_inputs: torch.Tensor) -> torch.Tensor:
        """Embed categorical inputs and concatenate with numerical inputs."""
        embedded = [emb(cat) for emb, cat in zip(self.embeddings, cat_inputs)]
        return torch.cat(embedded + [num_inputs], dim=-1)

    def encode(self, cat_inputs: List[torch.Tensor], num_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequence to latent distribution parameters.

        Uses the final hidden states from both directions of the BiLSTM,
        concatenated and projected to mu and logvar.
        """
        x = self._embed_inputs(cat_inputs, num_inputs)
        _, (h, _) = self.encoder(x)
        # h shape: (num_layers * 2, batch, encoder_hidden)
        # Take last layer forward and backward, concatenate
        h_fwd = h[-2]  # last layer, forward
        h_bwd = h[-1]  # last layer, backward
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (batch, 2 * encoder_hidden)
        return self.fc_mu(h_cat), self.fc_logvar(h_cat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode_teacher_forced(self, z: torch.Tensor, cat_inputs: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Teacher-forced decode (training). Parallel LSTM forward.

        The decoder sees the ground-truth previous activity at each position,
        shifted right by one (position 0 gets padding=0 as "start token").
        """
        n_steps = cat_inputs[0].size(1)
        batch = z.size(0)

        # Shift activities right by 1: [0, a0, a1, ..., a_{T-2}]
        prev_activities = torch.zeros_like(cat_inputs[0])
        prev_activities[:, 1:] = cat_inputs[0][:, :-1]

        prev_emb = self.decoder_embedding(prev_activities)  # (batch, n_steps, embed_dim)
        z_exp = z.unsqueeze(1).expand(-1, n_steps, -1)
        pos = self.pos_encoding[:n_steps].unsqueeze(0).expand(batch, -1, -1)

        decoder_input = torch.cat([z_exp, pos, prev_emb], dim=-1)
        out, _ = self.decoder(decoder_input)
        out = self.dec_dropout(out)

        cat_logits = [head(out) for head in self.cat_heads]
        num_out = self.num_head(out) if self.num_head else None
        return cat_logits, num_out

    def decode(self, z: torch.Tensor, n_steps: int = None) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Autoregressive decode from latent vector (inference).

        Step-by-step: each position uses the argmax prediction from the
        previous position as input. Early stopping: once padding (0) is
        predicted, all subsequent positions are forced to padding.
        """
        if n_steps is None:
            n_steps = self.seq_len

        batch = z.size(0)
        device = z.device

        prev_activity = torch.zeros(batch, dtype=torch.long, device=device)
        done = torch.zeros(batch, dtype=torch.bool, device=device)

        h_state = None
        all_cat_logits = [[] for _ in self.cat_heads]
        all_num_out = []

        for t in range(n_steps):
            prev_emb = self.decoder_embedding(prev_activity)
            pos = self.pos_encoding[t].unsqueeze(0).expand(batch, -1)
            step_input = torch.cat([z, pos, prev_emb], dim=-1).unsqueeze(1)

            out, h_state = self.decoder(step_input, h_state)
            out = out.squeeze(1)

            for i, head in enumerate(self.cat_heads):
                all_cat_logits[i].append(head(out))
            if self.num_head:
                all_num_out.append(self.num_head(out))

            # Next step: argmax of activity head
            prev_activity = self.cat_heads[0](out).argmax(dim=-1)
            # Early stopping: once padding predicted, stay in padding
            # (but not at t=0 — right-padded data always starts with a real event)
            prev_activity = prev_activity.masked_fill(done, 0)
            if t > 0:
                done = done | (prev_activity == 0)

        cat_logits = [torch.stack(logits_list, dim=1) for logits_list in all_cat_logits]
        num_out = torch.stack(all_num_out, dim=1) if all_num_out else None
        return cat_logits, num_out

    def decode_suffix(self, z: torch.Tensor, prefix_cat: List[torch.Tensor], prefix_num: torch.Tensor):
        """Decode suffix conditioned on prefix. Returns only the suffix portion."""
        prefix_len = prefix_cat[0].size(1)
        cat_logits, num_out = self.decode(z)
        suffix_cat = [l[:, prefix_len:, :] for l in cat_logits]
        suffix_num = num_out[:, prefix_len:, :] if num_out is not None else None
        return suffix_cat, suffix_num

    def generate_counterfactual(self, cat_inputs, num_inputs, prefix_len: int,
                                 n_samples: int = 1, noise_scale: float = 0.5):
        """Generate counterfactuals preserving prefix."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(cat_inputs, num_inputs)
            prefix_cat = [c[:, :prefix_len] for c in cat_inputs]
            prefix_num = num_inputs[:, :prefix_len]
            results = []
            for _ in range(n_samples):
                z = mu + noise_scale * torch.randn_like(mu)
                suffix_logits, suffix_num = self.decode_suffix(z, prefix_cat, prefix_num)
                suffix_preds = [l.argmax(dim=-1) for l in suffix_logits]
                full_cat = [torch.cat([prefix_cat[i], suffix_preds[i]], dim=1) for i in range(len(cat_inputs))]
                full_num = torch.cat([prefix_num, suffix_num], dim=1) if suffix_num is not None else None
                results.append((full_cat, full_num))
            return results

    def forward(self, cat_inputs: List[torch.Tensor], num_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Teacher-forced forward pass (training)."""
        mu, logvar = self.encode(cat_inputs, num_inputs)
        z = self.reparameterize(mu, logvar)
        cat_logits, num_out = self._decode_teacher_forced(z, cat_inputs)
        return {'mu': mu, 'logvar': logvar, 'z': z, 'cat_logits': cat_logits, 'num_out': num_out}

    def loss(
        self,
        cat_inputs: List[torch.Tensor],
        num_inputs: torch.Tensor,
        outputs: Dict,
        kl_weight: float = 1.0,
        constraints: Optional[List[DeclareConstraint]] = None,
        constraint_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss = reconstruction + KL + constraint penalty.

        CE loss on event positions + a few padding positions.
        - Event positions: learn correct activity predictions
        - First padding pos: learn "transition to stop" (prev=last_event → predict 0)
        - Next few padding pos: learn "stay stopped" (prev=0 → predict 0)
        - Remaining padding: ignored (avoids padding dominance)
        """
        device = cat_inputs[0].device
        IGNORE = -100
        PAD_CONTEXT = 5  # how many padding positions to include

        # Reconstruction loss for categoricals — masked CE
        recon_cat = torch.tensor(0.0, device=device)
        for logits, target in zip(outputs['cat_logits'], cat_inputs):
            target_masked = target.clone().long()
            for b in range(target.size(0)):
                event_len = int((target[b] != 0).sum().item())
                cutoff = min(event_len + PAD_CONTEXT, target.size(1))
                if cutoff < target.size(1):
                    target_masked[b, cutoff:] = IGNORE
            recon_cat = recon_cat + F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_masked.reshape(-1),
                ignore_index=IGNORE,
            )
        recon_cat = recon_cat / len(cat_inputs)

        # Reconstruction loss for numericals (masked to non-padding positions)
        if outputs['num_out'] is not None:
            mask = cat_inputs[0] != 0
            if mask.sum() > 0:
                recon_num = F.mse_loss(outputs['num_out'][mask], num_inputs[mask])
            else:
                recon_num = torch.tensor(0.0, device=device)
        else:
            recon_num = torch.tensor(0.0, device=device)

        # KL divergence
        kl = -0.5 * torch.mean(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())

        # Constraint penalty (non-differentiable, disabled by default)
        if constraints and constraint_weight > 0:
            activity_preds = outputs['cat_logits'][0].argmax(dim=-1)
            total_violations = 0
            for i in range(activity_preds.size(0)):
                seq = [x.item() for x in activity_preds[i] if x.item() != 0]
                total_violations += DeclareConstraintChecker.count_violations(seq, constraints)
            constraint_penalty = torch.tensor(total_violations / activity_preds.size(0), device=device)
        else:
            constraint_penalty = torch.tensor(0.0, device=device)

        total = recon_cat + recon_num + kl_weight * kl + constraint_weight * constraint_penalty
        return {
            'total': total,
            'recon_cat': recon_cat,
            'recon_num': recon_num,
            'kl': kl,
            'constraint': constraint_penalty,
        }

    def reconstruct(self, cat_inputs: List[torch.Tensor], num_inputs: torch.Tensor) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Encode then decode (autoregressive)."""
        with torch.no_grad():
            mu, _ = self.encode(cat_inputs, num_inputs)
            cat_logits, num_out = self.decode(mu)
            cat_preds = [logits.argmax(dim=-1) for logits in cat_logits]
            return cat_preds, num_out

    def sample(self, n: int, device: torch.device) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Sample from prior (autoregressive decode)."""
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            cat_logits, num_out = self.decode(z)
            cat_preds = [logits.argmax(dim=-1) for logits in cat_logits]
            return cat_preds, num_out


def train_vae(
    vae: SimpleSequenceVAE,
    dataloader,
    epochs: int = 100,
    lr: float = 1e-3,
    kl_weight: float = 0.1,
    constraints: Optional[List[DeclareConstraint]] = None,
    constraint_weight: float = 1.0,
    device: str = 'cpu',
    verbose: bool = True,
    grad_clip: float = 0.25,
    weight_decay: float = 1e-5,
) -> List[Dict[str, float]]:
    """Training loop with gradient clipping and cosine LR schedule."""
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 10
    )
    from tqdm.auto import tqdm
    history = []

    n_batches_total = len(dataloader)
    pbar = tqdm(range(epochs), desc="Training VAE", disable=not verbose)
    for epoch in pbar:
        vae.train()
        epoch_losses = {'total': 0, 'recon_cat': 0, 'recon_num': 0, 'kl': 0, 'constraint': 0}
        n_batches = 0

        batch_pbar = tqdm(
            dataloader, desc=f"  Epoch {epoch+1}/{epochs}",
            leave=False, disable=not verbose,
            total=n_batches_total,
        )
        for cat_inputs, num_inputs in batch_pbar:
            cat_inputs = [c.to(device) for c in cat_inputs]
            num_inputs = num_inputs.to(device)

            optimizer.zero_grad()
            outputs = vae(cat_inputs, num_inputs)
            loss_dict = vae.loss(
                cat_inputs, num_inputs, outputs,
                kl_weight=kl_weight,
                constraints=constraints,
                constraint_weight=constraint_weight,
            )
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
            optimizer.step()

            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key].item()
            n_batches += 1

            if n_batches % 10 == 0:
                batch_pbar.set_postfix_str(f"loss={loss_dict['total'].item():.4f}")

        scheduler.step()

        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        history.append(epoch_losses)

        loss_msg = f"loss={epoch_losses['total']:.4f}"
        if constraints and constraint_weight > 0:
            loss_msg += f", viol={epoch_losses['constraint']:.2f}"
        pbar.set_postfix_str(loss_msg)

    return history
