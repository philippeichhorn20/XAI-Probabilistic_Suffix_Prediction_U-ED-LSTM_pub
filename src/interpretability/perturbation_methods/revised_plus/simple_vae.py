"""
LSTM-VAE for process sequences.

Handles multiple categorical features (via embeddings) and numerical features.
Uses a parallel decoder (z repeated at each position) where the LSTM's
recurrent hidden state provides implicit positional/sequential context.
Training and inference use the same decoding path — no exposure bias.

Includes Declare constraint penalties using the existing DeclareConstraintChecker.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

# Import existing constraint infrastructure
from ..declare import DeclareConstraint, DeclareConstraintChecker


class SimpleSequenceVAE(nn.Module):
    """LSTM-VAE for process sequences with parallel z-only decoding."""

    def __init__(
        self,
        cat_vocab_sizes: List[int],
        num_features: int,
        seq_len: int,
        embed_dim: int = 24,
        hidden_dim: int = 96,
        latent_dim: int = 24,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cat_vocab_sizes = cat_vocab_sizes
        self.num_features = num_features

        # Embeddings for categorical features (padding_idx=0)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for vocab_size in cat_vocab_sizes
        ])

        self.input_dim = len(cat_vocab_sizes) * embed_dim + num_features

        # Encoder
        self.encoder = nn.LSTM(
            self.input_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z repeated at each position → LSTM
        # The LSTM hidden state provides implicit positional awareness.
        self.decoder = nn.LSTM(
            latent_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output heads
        self.cat_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for vocab_size in cat_vocab_sizes
        ])
        self.num_head = nn.Linear(hidden_dim, num_features) if num_features > 0 else None

    def _embed_inputs(self, cat_inputs: List[torch.Tensor], num_inputs: torch.Tensor) -> torch.Tensor:
        """Embed categorical inputs and concatenate with numerical inputs."""
        embedded = [emb(cat) for emb, cat in zip(self.embeddings, cat_inputs)]
        return torch.cat(embedded + [num_inputs], dim=-1)

    def encode(self, cat_inputs: List[torch.Tensor], num_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequence to latent distribution parameters."""
        x = self._embed_inputs(cat_inputs, num_inputs)
        _, (h, _) = self.encoder(x)
        h = h[-1]  # Last layer's hidden state
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, n_steps: int = None) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Decode from latent vector by repeating z at each position.

        Args:
            z: Latent vector (batch, latent_dim).
            n_steps: Number of timesteps (defaults to seq_len).

        Returns:
            cat_logits: List of (batch, n_steps, vocab_size) tensors.
            num_out: (batch, n_steps, num_features) tensor or None.
        """
        if n_steps is None:
            n_steps = self.seq_len

        z_exp = z.unsqueeze(1).expand(-1, n_steps, -1)
        out, _ = self.decoder(z_exp)

        cat_logits = [head(out) for head in self.cat_heads]
        num_out = self.num_head(out) if self.num_head else None
        return cat_logits, num_out

    def decode_suffix(self, z: torch.Tensor, prefix_cat: List[torch.Tensor], prefix_num: torch.Tensor):
        """Decode suffix conditioned on prefix.

        Runs full decode and returns only the suffix portion.
        """
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
        """Full forward pass — same parallel decode used at train and eval time."""
        mu, logvar = self.encode(cat_inputs, num_inputs)
        z = self.reparameterize(mu, logvar)
        cat_logits, num_out = self.decode(z)
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
        """Compute VAE loss = reconstruction + KL + constraint penalty."""
        device = cat_inputs[0].device

        # Reconstruction loss for categoricals (including padding positions,
        # so the model learns to predict padding where appropriate)
        recon_cat = sum(
            F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
            for logits, target in zip(outputs['cat_logits'], cat_inputs)
        ) / len(cat_inputs)

        # Reconstruction loss for numericals
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

        # Constraint penalty using existing DeclareConstraintChecker
        if constraints:
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
        """Encode then decode."""
        with torch.no_grad():
            mu, _ = self.encode(cat_inputs, num_inputs)
            cat_logits, num_out = self.decode(mu)
            cat_preds = [logits.argmax(dim=-1) for logits in cat_logits]
            return cat_preds, num_out

    def sample(self, n: int, device: torch.device) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Sample from prior."""
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
    grad_clip: float = 1.0,
) -> List[Dict[str, float]]:
    """Training loop with gradient clipping and cosine LR schedule."""
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 10
    )
    history = []

    for epoch in range(epochs):
        vae.train()
        epoch_losses = {'total': 0, 'recon_cat': 0, 'recon_num': 0, 'kl': 0, 'constraint': 0}
        n_batches = 0

        for cat_inputs, num_inputs in dataloader:
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

        scheduler.step()

        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        history.append(epoch_losses)

        if verbose and (epoch + 1) % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            msg = f"Epoch {epoch+1}/{epochs}, Loss: {epoch_losses['total']:.4f}, LR: {lr_now:.6f}"
            if constraints:
                msg += f", Violations: {epoch_losses['constraint']:.2f}"
            print(msg)

    return history
