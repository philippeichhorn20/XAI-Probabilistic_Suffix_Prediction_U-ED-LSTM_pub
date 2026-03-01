"""
Prefix encoder for LORELEY.

Converts process prefixes into fixed-length feature vectors suitable for
the genetic algorithm and decision tree components.
"""

import numpy as np
import torch
from typing import List, Tuple


class PrefixEncoder:
    """
    Encodes process prefixes into fixed-length feature vectors for LORELEY.

    Encoding scheme:
    - Control flow: frequency of each activity (bag-of-activities)
    - Numerical features: summary statistics (mean, sum, max, min, std)
    - Categorical features: one-hot encoding of unique values

    The control flow features are treated as a unit during genetic operations
    to maintain process structure.
    """

    def __init__(self,
                 activity_vocab_size: int,
                 num_feature_names: List[str],
                 cat_feature_names: List[str],
                 cat_vocab_sizes: List[int]):
        """
        Args:
            activity_vocab_size: Number of unique activities
            num_feature_names: Names of numerical features
            cat_feature_names: Names of categorical features (excluding Activity)
            cat_vocab_sizes: Vocabulary size for each categorical feature
        """
        self.activity_vocab_size = activity_vocab_size
        self.num_feature_names = num_feature_names
        self.cat_feature_names = cat_feature_names
        self.cat_vocab_sizes = cat_vocab_sizes

        # Build feature name mapping
        self.feature_names = self._build_feature_names()
        self.n_features = len(self.feature_names)

        # Indices for different feature types
        self._control_flow_end = activity_vocab_size
        self._num_features_end = self._control_flow_end + len(num_feature_names) * 5

    def _build_feature_names(self) -> List[str]:
        """Build list of feature names for encoded vector."""
        names = []

        # Control flow features (activity frequencies)
        for i in range(self.activity_vocab_size):
            names.append(f"activity_{i}_freq")

        # Numerical feature statistics
        for feat_name in self.num_feature_names:
            for stat in ['mean', 'sum', 'max', 'min', 'std']:
                names.append(f"{feat_name}_{stat}")

        # Categorical features (one-hot)
        for feat_name, vocab_size in zip(self.cat_feature_names, self.cat_vocab_sizes):
            for i in range(vocab_size):
                names.append(f"{feat_name}_{i}")

        return names

    def encode(self, prefix: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> np.ndarray:
        """
        Encode a prefix into a fixed-length feature vector.

        Args:
            prefix: Tuple of (cat_tensors, num_tensors)
                   cat_tensors[0] is Activity with shape [batch, seq_len]

        Returns:
            np.ndarray of shape [n_features]
        """
        cat_tensors, num_tensors = prefix

        # Assume batch size 1
        features = []

        # 1. Control flow: activity frequencies
        activity_seq = cat_tensors[0][0].cpu().numpy()  # [seq_len]
        freq = np.zeros(self.activity_vocab_size)
        for act in activity_seq:
            if 0 <= act < self.activity_vocab_size:
                freq[int(act)] += 1
        features.extend(freq)

        # 2. Numerical features: summary statistics
        for num_tensor in num_tensors:
            vals = num_tensor[0].cpu().numpy()  # [seq_len]
            features.append(np.mean(vals))
            features.append(np.sum(vals))
            features.append(np.max(vals))
            features.append(np.min(vals))
            features.append(np.std(vals))

        # 3. Categorical features (excluding Activity which is index 0)
        for i, cat_tensor in enumerate(cat_tensors[1:]):  # Skip Activity
            cat_seq = cat_tensor[0].cpu().numpy()  # [seq_len]
            vocab_size = self.cat_vocab_sizes[i]
            one_hot = np.zeros(vocab_size)
            # Use last value as representative (or could use mode)
            last_val = int(cat_seq[-1])
            if 0 <= last_val < vocab_size:
                one_hot[last_val] = 1
            features.extend(one_hot)

        return np.array(features, dtype=np.float32)

    def get_control_flow_indices(self) -> List[int]:
        """Return indices of control flow features."""
        return list(range(self._control_flow_end))

    def get_numerical_indices(self) -> List[int]:
        """Return indices of numerical features."""
        return list(range(self._control_flow_end, self._num_features_end))

    def get_categorical_indices(self) -> List[int]:
        """Return indices of categorical features."""
        return list(range(self._num_features_end, self.n_features))

    def get_control_flow_features(self, encoded: np.ndarray) -> np.ndarray:
        """Extract control flow features from encoded vector."""
        return encoded[:self._control_flow_end]

    def get_numerical_features(self, encoded: np.ndarray) -> np.ndarray:
        """Extract numerical features from encoded vector."""
        return encoded[self._control_flow_end:self._num_features_end]

    def get_categorical_features(self, encoded: np.ndarray) -> np.ndarray:
        """Extract categorical features from encoded vector."""
        return encoded[self._num_features_end:]
