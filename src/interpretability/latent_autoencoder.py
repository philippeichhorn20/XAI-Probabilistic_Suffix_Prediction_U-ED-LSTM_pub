"""
Latent Autoencoder for learning feature representations.

This module provides an autoencoder that learns compressed representations
of the input features (categorical embeddings + numerical values) from
process event sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
from tqdm.auto import tqdm


@dataclass
class AutoencoderConfig:
    """Configuration for the latent autoencoder."""
    # Input dimensions (set automatically from data)
    cat_vocab_sizes: List[int] = None  # Number of classes for each categorical feature
    cat_embedding_dims: List[int] = None  # Embedding dimension for each categorical feature
    num_features: int = 0  # Number of numerical features

    # Architecture
    latent_dim: int = 32  # Dimension of latent space
    hidden_dims: List[int] = None  # Hidden layer dimensions [128, 64]

    # Training
    dropout: float = 0.1
    use_batch_norm: bool = True

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


class FeatureEncoder(nn.Module):
    """Encoder: maps input features to latent space."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim)
            for vocab_size, emb_dim in zip(config.cat_vocab_sizes, config.cat_embedding_dims)
        ])

        # Calculate total input dimension
        total_cat_dim = sum(config.cat_embedding_dims)
        input_dim = total_cat_dim + config.num_features

        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        # Final projection to latent space
        layers.append(nn.Linear(prev_dim, config.latent_dim))

        self.encoder = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, cat_inputs: List[torch.Tensor], num_inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode inputs to latent space.

        Args:
            cat_inputs: List of categorical tensors, each (batch,) or (batch, seq_len)
            num_inputs: Numerical tensor (batch, num_features) or (batch, seq_len, num_features)

        Returns:
            Latent representation (batch, latent_dim) or (batch, seq_len, latent_dim)
        """
        # Get embeddings for categorical features
        cat_embeds = []
        for i, (emb_layer, cat_tensor) in enumerate(zip(self.cat_embeddings, cat_inputs)):
            embedded = emb_layer(cat_tensor)  # (batch, emb_dim) or (batch, seq_len, emb_dim)
            cat_embeds.append(embedded)

        # Concatenate all features
        cat_concat = torch.cat(cat_embeds, dim=-1)

        if num_inputs is not None and num_inputs.numel() > 0:
            features = torch.cat([cat_concat, num_inputs], dim=-1)
        else:
            features = cat_concat

        # Handle sequence dimension if present (3D input)
        original_shape = features.shape
        if features.dim() == 3:
            batch, seq_len, feat_dim = features.shape
            features = features.view(batch * seq_len, feat_dim)

        # Encode
        latent = self.encoder(features)

        # Restore sequence dimension if needed
        if len(original_shape) == 3:
            latent = latent.view(batch, seq_len, -1)

        return latent


class FeatureDecoder(nn.Module):
    """Decoder: reconstructs input features from latent space."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config

        # Calculate total output dimension
        total_cat_dim = sum(config.cat_embedding_dims)
        output_dim = total_cat_dim + config.num_features

        # Build decoder layers (reverse of encoder)
        layers = []
        prev_dim = config.latent_dim
        for hidden_dim in reversed(config.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        self.decoder = nn.Sequential(*layers)

        # Separate output heads for categorical and numerical features
        self.cat_output_heads = nn.ModuleList([
            nn.Linear(prev_dim, vocab_size)
            for vocab_size in config.cat_vocab_sizes
        ])

        self.num_output_head = nn.Linear(prev_dim, config.num_features) if config.num_features > 0 else None

    def forward(self, latent: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Decode latent representation to reconstructed features.

        Args:
            latent: Latent tensor (batch, latent_dim) or (batch, seq_len, latent_dim)

        Returns:
            cat_logits: List of logits for each categorical feature
            num_output: Reconstructed numerical features
        """
        original_ndim = latent.dim()
        if latent.dim() == 3:
            batch, seq_len, latent_dim = latent.shape
            latent = latent.view(batch * seq_len, latent_dim)

        # Decode
        hidden = self.decoder(latent)

        # Get categorical logits
        cat_logits = [head(hidden) for head in self.cat_output_heads]

        # Get numerical output
        num_output = self.num_output_head(hidden) if self.num_output_head is not None else None

        # Restore sequence dimension if needed
        if original_ndim == 3:
            cat_logits = [logits.view(batch, seq_len, -1) for logits in cat_logits]
            if num_output is not None:
                num_output = num_output.view(batch, seq_len, -1)

        return cat_logits, num_output


class LatentAutoencoder(nn.Module):
    """
    Autoencoder for learning latent representations of process event features.

    The model encodes categorical (via embeddings) and numerical features into
    a compressed latent space, then reconstructs them.
    """

    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        self.encoder = FeatureEncoder(config)
        self.decoder = FeatureDecoder(config)

    def forward(
        self,
        cat_inputs: List[torch.Tensor],
        num_inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through autoencoder.

        Returns:
            latent: Latent representation
            cat_logits: Reconstructed categorical logits
            num_output: Reconstructed numerical values
        """
        latent = self.encoder(cat_inputs, num_inputs)
        cat_logits, num_output = self.decoder(latent)
        return latent, cat_logits, num_output

    def encode(self, cat_inputs: List[torch.Tensor], num_inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs to latent space."""
        return self.encoder(cat_inputs, num_inputs)

    def decode(self, latent: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Decode latent representation."""
        return self.decoder(latent)

    def compute_loss(
        self,
        cat_inputs: List[torch.Tensor],
        num_inputs: torch.Tensor,
        cat_logits: List[torch.Tensor],
        num_output: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss.

        Returns:
            Dictionary with 'total', 'categorical', 'numerical' losses
        """
        device = cat_logits[0].device if cat_logits else num_output.device
        losses = {}

        # Categorical reconstruction loss (cross-entropy)
        cat_loss = torch.tensor(0.0, device=device)
        for i, (target, logits) in enumerate(zip(cat_inputs, cat_logits)):
            # Flatten for cross-entropy if needed
            if logits.dim() == 3:
                logits_flat = logits.view(-1, logits.size(-1))
                target_flat = target.view(-1)
            elif logits.dim() == 2:
                logits_flat = logits
                target_flat = target
            else:
                continue

            # Ignore padding (index 0)
            mask = target_flat != 0
            if mask.sum() > 0:
                cat_loss = cat_loss + F.cross_entropy(logits_flat[mask], target_flat[mask])

        cat_loss = cat_loss / len(cat_inputs) if cat_inputs else cat_loss
        losses['categorical'] = cat_loss

        # Numerical reconstruction loss (MSE)
        if num_output is not None and num_inputs is not None and num_inputs.numel() > 0:
            # Create mask for non-padding positions (where first categorical is non-zero)
            if cat_inputs:
                mask = cat_inputs[0] != 0
                # Expand mask to match numerical dimensions
                if mask.dim() == 1 and num_inputs.dim() == 2:
                    mask = mask.unsqueeze(-1).expand_as(num_inputs)
                elif mask.dim() == 2 and num_inputs.dim() == 3:
                    mask = mask.unsqueeze(-1).expand_as(num_inputs)

                if mask.sum() > 0:
                    num_loss = F.mse_loss(num_output[mask], num_inputs[mask])
                else:
                    num_loss = torch.tensor(0.0, device=device)
            else:
                num_loss = F.mse_loss(num_output, num_inputs)
            losses['numerical'] = num_loss
        else:
            losses['numerical'] = torch.tensor(0.0, device=device)

        # Total loss
        losses['total'] = losses['categorical'] + losses['numerical']

        return losses

    @classmethod
    def from_model_categories(
        cls,
        data_set_categories: Tuple,
        latent_dim: int = 32,
        hidden_dims: List[int] = None,
        **kwargs
    ) -> 'LatentAutoencoder':
        """
        Create autoencoder from model's data_set_categories.

        Args:
            data_set_categories: Tuple of (cat_categories, num_categories) from model
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions
            **kwargs: Additional config options
        """
        cat_categories, num_categories = data_set_categories

        # Extract vocab sizes and compute embedding dims
        cat_vocab_sizes = [size for _, size, _ in cat_categories]
        cat_embedding_dims = [min(50, max(4, int(np.ceil(np.log2(size) * 2)))) for size in cat_vocab_sizes]
        num_features = len(num_categories)

        config = AutoencoderConfig(
            cat_vocab_sizes=cat_vocab_sizes,
            cat_embedding_dims=cat_embedding_dims,
            num_features=num_features,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims or [128, 64],
            **kwargs
        )

        return cls(config)


class AutoencoderTrainer:
    """Trainer for the latent autoencoder."""

    def __init__(
        self,
        model: LatentAutoencoder,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.history = {'train_loss': [], 'val_loss': []}

    def _unpack_batch(self, batch):
        """Unpack batch into categorical and numerical inputs."""
        # Last tensor is numerical, rest are categorical
        cat_inputs = [t.to(self.device) for t in batch[:-1]]
        num_inputs = batch[-1].to(self.device)
        return cat_inputs, num_inputs

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            cat_inputs, num_inputs = self._unpack_batch(batch)

            self.optimizer.zero_grad()
            latent, cat_logits, num_output = self.model(cat_inputs, num_inputs)
            losses = self.model.compute_loss(cat_inputs, num_inputs, cat_logits, num_output)

            losses['total'].backward()
            self.optimizer.step()

            total_loss += losses['total'].item()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0

        for batch in dataloader:
            cat_inputs, num_inputs = self._unpack_batch(batch)

            latent, cat_logits, num_output = self.model(cat_inputs, num_inputs)
            losses = self.model.compute_loss(cat_inputs, num_inputs, cat_logits, num_output)

            total_loss += losses['total'].item()

        return total_loss / len(dataloader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train the autoencoder.

        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

        for epoch in iterator:
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if verbose:
                    iterator.set_postfix({
                        'train': f'{train_loss:.4f}',
                        'val': f'{val_loss:.4f}'
                    })

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            else:
                if verbose:
                    iterator.set_postfix({'train': f'{train_loss:.4f}'})

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history


def prepare_dataloader(
    dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    flatten_sequences: bool = True
) -> DataLoader:
    """
    Prepare a DataLoader from the test dataset.

    Args:
        dataset: Test dataset with (cat_tensors, num_tensors, case_name) samples
        batch_size: Batch size
        shuffle: Whether to shuffle
        flatten_sequences: If True, flatten sequences to individual events

    Returns:
        DataLoader
    """
    # Collect all events as rows
    all_events = []  # List of (cat_values, num_values) tuples

    for sample in dataset:
        cat_tensors, num_tensors, _ = sample

        # Get sequence length from first categorical tensor
        cat_t = cat_tensors[0]
        if cat_t.dim() > 1:
            cat_t = cat_t.squeeze(0)

        # Find non-padding positions
        mask = cat_t != 0

        for pos in range(len(cat_t)):
            if not mask[pos]:
                continue

            # Extract categorical values at this position
            cat_vals = []
            for c_tensor in cat_tensors:
                if c_tensor.dim() > 1:
                    c_tensor = c_tensor.squeeze(0)
                cat_vals.append(c_tensor[pos].item())

            # Extract numerical values at this position
            num_vals = []
            for n_tensor in num_tensors:
                if n_tensor.dim() > 1:
                    n_tensor = n_tensor.squeeze(0)
                num_vals.append(n_tensor[pos].item())

            all_events.append((cat_vals, num_vals))

    # Convert to tensors
    n_events = len(all_events)
    n_cat = len(all_events[0][0])
    n_num = len(all_events[0][1])

    cat_tensors = [torch.zeros(n_events, dtype=torch.long) for _ in range(n_cat)]
    num_tensor = torch.zeros(n_events, n_num, dtype=torch.float)

    for i, (cat_vals, num_vals) in enumerate(all_events):
        for j, val in enumerate(cat_vals):
            cat_tensors[j][i] = val
        for j, val in enumerate(num_vals):
            num_tensor[i, j] = val

    # Create dataset
    tensors = cat_tensors + [num_tensor]
    tensor_dataset = TensorDataset(*tensors)

    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
