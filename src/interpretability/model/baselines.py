"""
Baseline generation strategies for Integrated Gradients.

The baseline represents the "absence of information" in the input.
Different baselines can lead to different attributions, so choosing
an appropriate baseline is important for meaningful explanations.

Common strategies:
- Zero embedding: All-zeros baseline (simple but may not be meaningful)
- Padding token: Use a special padding/unknown token embedding
- Mean embedding: Use the mean embedding from training data
- Uniform: Uniform distribution over vocabulary for categorical
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
import numpy as np


class BaselineGenerator(ABC):
    """Abstract base class for generating baselines."""

    @abstractmethod
    def generate(
        self,
        inputs: Tuple[List[Tensor], List[Tensor]],
        embedded_inputs: Tensor
    ) -> Tensor:
        """
        Generate baseline for the given inputs.

        Args:
            inputs: Original model inputs (categorical tensors, numerical tensors)
            embedded_inputs: Embedded representation of inputs

        Returns:
            Baseline tensor with same shape as embedded_inputs
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return description of the baseline strategy."""
        pass


class ZeroBaseline(BaselineGenerator):
    """
    Zero baseline: all-zeros tensor.

    Simple and commonly used, but may not represent a meaningful
    "absence of information" for all models.
    """

    def generate(
        self,
        inputs: Tuple[List[Tensor], List[Tensor]],
        embedded_inputs: Tensor
    ) -> Tensor:
        """Generate zero baseline."""
        return torch.zeros_like(embedded_inputs)

    def get_description(self) -> str:
        return "Zero embedding baseline"


class PaddingBaseline(BaselineGenerator):
    """
    Padding baseline: use the embedding of a padding/unknown token.

    More meaningful than zeros for models trained with padding tokens,
    as it represents the embedding of "no information".
    """

    def __init__(
        self,
        embeddings: nn.ModuleList,
        padding_indices: Optional[List[int]] = None,
        numerical_baseline: str = 'zero'
    ):
        """
        Initialize padding baseline generator.

        Args:
            embeddings: List of embedding modules from the model
            padding_indices: Index of padding token for each embedding.
                           Defaults to 0 for each.
            numerical_baseline: How to handle numerical features:
                              'zero', 'mean', or a specific value
        """
        self.embeddings = embeddings
        self.padding_indices = padding_indices or [0] * len(embeddings)
        self.numerical_baseline = numerical_baseline
        self._numerical_mean = None

    def set_numerical_mean(self, mean_values: List[float]):
        """Set mean values for numerical features."""
        self._numerical_mean = mean_values

    def generate(
        self,
        inputs: Tuple[List[Tensor], List[Tensor]],
        embedded_inputs: Tensor
    ) -> Tensor:
        """Generate padding baseline."""
        cat_inputs, num_inputs = inputs
        device = embedded_inputs.device
        batch_size = embedded_inputs.shape[1]  # [seq_len, batch, features]
        seq_len = embedded_inputs.shape[0]

        # Get padding embeddings for categorical features
        embedded_padding = []
        for i, (embedding, pad_idx) in enumerate(zip(self.embeddings, self.padding_indices)):
            pad_tensor = torch.full(
                (batch_size, seq_len),
                pad_idx,
                dtype=torch.long,
                device=device
            )
            pad_embed = embedding(pad_tensor)  # [batch, seq_len, embed_dim]
            pad_embed = pad_embed.permute(1, 0, 2)  # [seq_len, batch, embed_dim]
            embedded_padding.append(pad_embed)

        # Concatenate categorical embeddings
        cat_baseline = torch.cat(embedded_padding, dim=-1)

        # Handle numerical features
        if num_inputs:
            num_size = embedded_inputs.shape[-1] - cat_baseline.shape[-1]
            if self.numerical_baseline == 'zero':
                num_baseline = torch.zeros(
                    seq_len, batch_size, num_size,
                    device=device, dtype=embedded_inputs.dtype
                )
            elif self.numerical_baseline == 'mean' and self._numerical_mean is not None:
                mean_tensor = torch.tensor(
                    self._numerical_mean, device=device, dtype=embedded_inputs.dtype
                )
                num_baseline = mean_tensor.unsqueeze(0).unsqueeze(0).expand(
                    seq_len, batch_size, -1
                )
            else:
                num_baseline = torch.zeros(
                    seq_len, batch_size, num_size,
                    device=device, dtype=embedded_inputs.dtype
                )

            baseline = torch.cat([cat_baseline, num_baseline], dim=-1)
        else:
            baseline = cat_baseline

        return baseline

    def get_description(self) -> str:
        return f"Padding token baseline (numerical: {self.numerical_baseline})"


class MeanBaseline(BaselineGenerator):
    """
    Mean baseline: use the mean embedding computed from training data.

    Represents the "average" input and can provide more meaningful
    attributions for understanding what makes an input different from average.
    """

    def __init__(self):
        """Initialize mean baseline generator."""
        self._mean_embedding = None
        self._computed = False

    def compute_mean(
        self,
        dataloader,
        model,
        max_samples: int = 1000
    ):
        """
        Compute mean embedding from training data.

        Args:
            dataloader: DataLoader with training data
            model: Model with embedding layers
            max_samples: Maximum number of samples to use
        """
        model.eval()
        all_embeddings = []
        n_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                if n_samples >= max_samples:
                    break

                prefixes = batch[:2]  # (cat_tensors, num_tensors)

                # Get embeddings through encoder's data preparation
                embedded = model.encoder.data_enc_for_model(
                    data=prefixes
                )
                all_embeddings.append(embedded.mean(dim=(0, 1)))  # Average over seq and batch

                n_samples += embedded.shape[1]

        # Compute overall mean
        self._mean_embedding = torch.stack(all_embeddings).mean(dim=0)
        self._computed = True

    def set_mean(self, mean_embedding: Tensor):
        """Manually set the mean embedding."""
        self._mean_embedding = mean_embedding
        self._computed = True

    def generate(
        self,
        inputs: Tuple[List[Tensor], List[Tensor]],
        embedded_inputs: Tensor
    ) -> Tensor:
        """Generate mean baseline."""
        if not self._computed or self._mean_embedding is None:
            raise RuntimeError(
                "Mean baseline not computed. Call compute_mean() or set_mean() first."
            )

        # Broadcast mean to match input shape
        baseline = self._mean_embedding.unsqueeze(0).unsqueeze(0).expand_as(embedded_inputs)
        return baseline.clone()

    def get_description(self) -> str:
        return "Mean embedding baseline"


class UniformDistributionBaseline(BaselineGenerator):
    """
    Uniform distribution baseline for categorical features.

    Instead of using a specific token embedding, uses the average of all
    token embeddings, representing maximum uncertainty.
    """

    def __init__(
        self,
        embeddings: nn.ModuleList,
        numerical_baseline: str = 'zero'
    ):
        """
        Initialize uniform baseline generator.

        Args:
            embeddings: List of embedding modules
            numerical_baseline: How to handle numerical features
        """
        self.embeddings = embeddings
        self.numerical_baseline = numerical_baseline
        self._numerical_mean = None

    def set_numerical_mean(self, mean_values: List[float]):
        """Set mean values for numerical features."""
        self._numerical_mean = mean_values

    def generate(
        self,
        inputs: Tuple[List[Tensor], List[Tensor]],
        embedded_inputs: Tensor
    ) -> Tensor:
        """Generate uniform distribution baseline."""
        device = embedded_inputs.device
        seq_len, batch_size, _ = embedded_inputs.shape

        # Compute mean of all embeddings for each categorical feature
        embedded_uniform = []
        for embedding in self.embeddings:
            # Get all embeddings and compute mean
            all_embeds = embedding.weight  # [vocab_size, embed_dim]
            mean_embed = all_embeds.mean(dim=0)  # [embed_dim]

            # Expand to match input shape
            uniform = mean_embed.unsqueeze(0).unsqueeze(0).expand(
                seq_len, batch_size, -1
            )
            embedded_uniform.append(uniform)

        # Concatenate categorical baselines
        cat_baseline = torch.cat(embedded_uniform, dim=-1)

        # Handle numerical features
        num_inputs = inputs[1]
        if num_inputs:
            num_size = embedded_inputs.shape[-1] - cat_baseline.shape[-1]
            if self.numerical_baseline == 'zero':
                num_baseline = torch.zeros(
                    seq_len, batch_size, num_size,
                    device=device, dtype=embedded_inputs.dtype
                )
            elif self.numerical_baseline == 'mean' and self._numerical_mean is not None:
                mean_tensor = torch.tensor(
                    self._numerical_mean, device=device, dtype=embedded_inputs.dtype
                )
                num_baseline = mean_tensor.unsqueeze(0).unsqueeze(0).expand(
                    seq_len, batch_size, -1
                )
            else:
                num_baseline = torch.zeros(
                    seq_len, batch_size, num_size,
                    device=device, dtype=embedded_inputs.dtype
                )

            baseline = torch.cat([cat_baseline, num_baseline], dim=-1)
        else:
            baseline = cat_baseline

        return baseline

    def get_description(self) -> str:
        return "Uniform distribution baseline (mean of all embeddings)"


class BlurredBaseline(BaselineGenerator):
    """
    Blurred baseline: average of embeddings in a local neighborhood.

    Uses a Gaussian blur over the sequence to create a "smoothed" baseline.
    Useful for understanding local vs global importance.
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        """
        Initialize blurred baseline generator.

        Args:
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation of the Gaussian
        """
        self.kernel_size = kernel_size
        self.sigma = sigma

    def generate(
        self,
        inputs: Tuple[List[Tensor], List[Tensor]],
        embedded_inputs: Tensor
    ) -> Tensor:
        """Generate blurred baseline."""
        # Create 1D Gaussian kernel
        x = torch.arange(self.kernel_size, dtype=embedded_inputs.dtype,
                        device=embedded_inputs.device)
        x = x - (self.kernel_size - 1) / 2
        kernel = torch.exp(-x**2 / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()

        # Apply blur to each feature dimension
        # embedded_inputs: [seq_len, batch, features]
        seq_len, batch_size, features = embedded_inputs.shape

        # Pad and apply convolution
        padding = self.kernel_size // 2
        padded = torch.nn.functional.pad(
            embedded_inputs.permute(1, 2, 0),  # [batch, features, seq_len]
            (padding, padding),
            mode='reflect'
        )

        # Convolve each feature
        blurred = torch.nn.functional.conv1d(
            padded,
            kernel.view(1, 1, -1).expand(features, 1, -1),
            groups=features
        )

        return blurred.permute(2, 0, 1)  # Back to [seq_len, batch, features]

    def get_description(self) -> str:
        return f"Blurred baseline (kernel={self.kernel_size}, sigma={self.sigma})"


def create_baseline_generator(
    strategy: str,
    embeddings: Optional[nn.ModuleList] = None,
    **kwargs
) -> BaselineGenerator:
    """
    Factory function to create baseline generators.

    Args:
        strategy: Baseline strategy name:
                 'zero', 'padding', 'mean', 'uniform', 'blurred'
        embeddings: Embedding modules (required for some strategies)
        **kwargs: Additional arguments for the strategy

    Returns:
        BaselineGenerator instance
    """
    if strategy == 'zero':
        return ZeroBaseline()
    elif strategy == 'padding':
        if embeddings is None:
            raise ValueError("Embeddings required for padding baseline")
        return PaddingBaseline(embeddings, **kwargs)
    elif strategy == 'mean':
        return MeanBaseline()
    elif strategy == 'uniform':
        if embeddings is None:
            raise ValueError("Embeddings required for uniform baseline")
        return UniformDistributionBaseline(embeddings, **kwargs)
    elif strategy == 'blurred':
        return BlurredBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy}")
