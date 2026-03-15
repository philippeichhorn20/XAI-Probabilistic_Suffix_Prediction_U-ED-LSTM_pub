"""
Decodes encoded tensor outputs back to human-readable categorical and numerical values.

Initialized from an EventLogDataset (which stores all_categories and encoder_decoder),
this class converts integer-encoded categorical indices back to original string values
and inverse-transforms standardized numerical features.

Works with any dataset — the mappings are read from the dataset object at init time.

Usage:
    dataset = torch.load('encoded_data/test_philipp/helpdesk_all_5_test.pkl', weights_only=False)
    decoder = TensorDecoder(dataset)

    # Decode a full sequence to a DataFrame
    cat_tuple, num_tuple, case_id = dataset[0]
    df = decoder.decode_sequence(cat_tuple, num_tuple, case_id=case_id)

    # Decode just an activity index list
    decoder.decode_categorical('Activity', [1, 13, 11, 2, 5])
    # -> ['Assign seriousness', 'Wait', 'Resolve ticket', 'EOS', 'Closed']

    # Decode a single index
    decoder.decode_categorical_value('Activity', 13)
    # -> 'Wait'
"""

from typing import Union, Optional
import torch
import numpy as np
import pandas as pd


class TensorDecoder:
    """
    Converts encoded tensors (from EventLogDataset) back to original values.

    Attributes:
        cat_features: list of categorical feature names in order
        num_features: list of numerical feature names in order
        idx_to_label: dict mapping feature_name -> {encoded_index: original_value}
        label_to_idx: dict mapping feature_name -> {original_value: encoded_index}
    """

    def __init__(self, dataset):
        """
        Args:
            dataset: An EventLogDataset instance (or any object with
                     .all_categories and .encoder_decoder attributes).
        """
        self.dataset = dataset
        all_categories = dataset.all_categories

        # --- categorical mappings ---
        # all_categories[0] is a list of (col_name, max_classes, {value: index})
        self.cat_features = [entry[0] for entry in all_categories[0]]
        self.label_to_idx = {
            entry[0]: entry[2] for entry in all_categories[0]
        }
        # invert: index -> label
        # Index 0 is padding for the first feature (activity). For other
        # features, 0 may represent a missing/unknown value rather than
        # padding, so label it accordingly.
        self.idx_to_label = {}
        for i, (feat_name, mapping) in enumerate(self.label_to_idx.items()):
            inverted = {v: k for k, v in mapping.items()}
            if 0 not in inverted:
                inverted[0] = "<pad>" if i == 0 else "<none>"
            self.idx_to_label[feat_name] = inverted

        # vocab sizes: max index + 1 per categorical feature (authoritative from encoder)
        self._vocab_sizes = []
        for feat_name in self.cat_features:
            max_idx = max(self.label_to_idx[feat_name].values()) if self.label_to_idx[feat_name] else 0
            self._vocab_sizes.append(max_idx + 1)  # +1 for 0-based indexing (padding at 0)

        # --- numerical feature names ---
        self.num_features = [entry[0] for entry in all_categories[1]]

        # --- reference to scalers for inverse-transforming numericals ---
        self.encoder_decoder = dataset.encoder_decoder

    # ------------------------------------------------------------------
    # Categorical decoding
    # ------------------------------------------------------------------

    def decode_categorical_value(self, feature: Union[str, int], index: int) -> Optional[str]:
        """Decode a single encoded index for a categorical feature.

        Args:
            feature: Feature name (str) or positional index (int) into cat_features.
            index: The encoded integer index.

        Returns:
            The original category string, or '<pad>' for padding (0).
        """
        feat_name = self._resolve_cat_feature(feature)
        return self.idx_to_label[feat_name].get(index, f"<unknown:{index}>")

    def decode_categorical(
        self,
        feature: Union[str, int],
        indices,
        skip_padding: bool = False,
    ) -> list:
        """Decode a list/tensor of encoded indices for one categorical feature.

        Args:
            feature: Feature name or positional index.
            indices: list, numpy array, or torch.Tensor of encoded integers.
            skip_padding: If True, omit entries where index == 0.

        Returns:
            List of decoded category strings.
        """
        feat_name = self._resolve_cat_feature(feature)
        mapping = self.idx_to_label[feat_name]

        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()

        result = []
        for idx in indices:
            if skip_padding and idx == 0:
                continue
            result.append(mapping.get(idx, f"<unknown:{idx}>"))
        return result

    # ------------------------------------------------------------------
    # Numerical decoding
    # ------------------------------------------------------------------

    def decode_numerical_value(self, feature: Union[str, int], value: float) -> float:
        """Inverse-transform a single standardized numerical value.

        Args:
            feature: Feature name or positional index into num_features.
            value: The standardized (z-scored) value.

        Returns:
            The original-scale value.
        """
        feat_name = self._resolve_num_feature(feature)
        scaler = self.encoder_decoder.continuous_encoders[feat_name]
        return scaler.inverse_transform([[value]])[0][0]

    def decode_numerical(
        self,
        feature: Union[str, int],
        values,
        skip_padding: bool = False,
    ) -> list:
        """Inverse-transform a sequence of standardized numerical values.

        Args:
            feature: Feature name or positional index.
            values: list, numpy array, or torch.Tensor of standardized values.
            skip_padding: If True, omit entries where the corresponding
                          categorical activity is padding (requires external check).

        Returns:
            List of original-scale floats.
        """
        feat_name = self._resolve_num_feature(feature)
        scaler = self.encoder_decoder.continuous_encoders[feat_name]

        if isinstance(values, torch.Tensor):
            values = values.numpy()
        if isinstance(values, list):
            values = np.array(values)

        arr = values.reshape(-1, 1)
        decoded = scaler.inverse_transform(arr).flatten()
        return decoded.tolist()

    # ------------------------------------------------------------------
    # Full-sequence decoding
    # ------------------------------------------------------------------

    def decode_sequence(
        self,
        cat_tuple,
        num_tuple=None,
        case_id=None,
        skip_padding: bool = True,
        activity_feature: Union[str, int] = 0,
    ) -> pd.DataFrame:
        """Decode a complete encoded sequence into a readable DataFrame.

        Args:
            cat_tuple: Tuple of categorical tensors, each of shape (seq_len,).
            num_tuple: Tuple of numerical tensors, each of shape (seq_len,).
                       Can be None if no numerical features.
            case_id: Optional case identifier to include in the DataFrame.
            skip_padding: If True, drop timesteps where the activity feature
                          is padding (0).
            activity_feature: Which categorical feature to use for padding
                              detection (default 0 = first = Activity).

        Returns:
            pd.DataFrame with one row per non-padding timestep and columns
            for each feature (using original human-readable values).
        """
        seq_len = cat_tuple[0].shape[0] if isinstance(cat_tuple[0], torch.Tensor) else len(cat_tuple[0])

        # Determine which positions are non-padding (based on activity feature)
        act_idx = self.cat_features.index(self._resolve_cat_feature(activity_feature)) if isinstance(activity_feature, str) else activity_feature
        act_cat = cat_tuple[act_idx]
        if isinstance(act_cat, torch.Tensor):
            act_cat = act_cat.tolist()
        mask = [i for i in range(seq_len) if (not skip_padding or act_cat[i] != 0)]

        data = {}

        # Categorical columns
        for feat_idx, feat_name in enumerate(self.cat_features):
            tensor = cat_tuple[feat_idx]
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.tolist()
            mapping = self.idx_to_label[feat_name]
            data[feat_name] = [mapping.get(tensor[i], f"<unknown:{tensor[i]}>") for i in mask]

        # Numerical columns
        if num_tuple is not None:
            for feat_idx, feat_name in enumerate(self.num_features):
                tensor = num_tuple[feat_idx]
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.numpy()
                elif isinstance(tensor, list):
                    tensor = np.array(tensor)
                # Inverse-transform all values, then select masked positions
                arr = tensor.reshape(-1, 1)
                scaler = self.encoder_decoder.continuous_encoders[feat_name]
                decoded = scaler.inverse_transform(arr).flatten()
                data[feat_name] = [decoded[i] for i in mask]

        if case_id is not None:
            data["Case ID"] = [case_id] * len(mask)

        return pd.DataFrame(data)

    def decode_batch(
        self,
        cat_batch: list,
        num_batch=None,
        case_ids=None,
        skip_padding: bool = True,
    ) -> list:
        """Decode a batch of sequences.

        Args:
            cat_batch: List of categorical tensors, each of shape (batch, seq_len).
            num_batch: Numerical tensor of shape (batch, seq_len, n_num), or None.
            case_ids: Optional list of case IDs.
            skip_padding: Drop padding timesteps.

        Returns:
            List of DataFrames, one per sequence in the batch.
        """
        batch_size = cat_batch[0].shape[0]
        results = []
        for b in range(batch_size):
            cat_tuple = tuple(c[b] for c in cat_batch)
            num_tuple = None
            if num_batch is not None:
                # num_batch shape: (batch, seq_len, n_num) -> split into per-feature tensors
                if isinstance(num_batch, torch.Tensor) and num_batch.dim() == 3:
                    num_tuple = tuple(num_batch[b, :, i] for i in range(num_batch.shape[2]))
                else:
                    # Already a list of tensors per feature
                    num_tuple = tuple(n[b] for n in num_batch)
            cid = case_ids[b] if case_ids is not None else None
            results.append(self.decode_sequence(cat_tuple, num_tuple, case_id=cid, skip_padding=skip_padding))
        return results

    # ------------------------------------------------------------------
    # Convenience / inspection
    # ------------------------------------------------------------------

    def get_vocab_sizes(self) -> list:
        """Return vocab sizes for all categorical features.

        These are derived from the authoritative encoder mappings, not from
        scanning data. Use these when constructing models (e.g., VAE embedding
        layers) to ensure all valid categories are covered.

        Returns:
            List of ints, one per categorical feature. Each is max_index + 1.
        """
        return list(self._vocab_sizes)

    def get_vocabulary(self, feature: Union[str, int] = 0) -> dict:
        """Return the full {label: index} vocabulary for a categorical feature."""
        feat_name = self._resolve_cat_feature(feature)
        return dict(self.label_to_idx[feat_name])

    def get_inverted_vocabulary(self, feature: Union[str, int] = 0) -> dict:
        """Return the full {index: label} vocabulary for a categorical feature."""
        feat_name = self._resolve_cat_feature(feature)
        return dict(self.idx_to_label[feat_name])

    def summary(self) -> str:
        """Return a human-readable summary of all features and their vocabularies."""
        lines = ["TensorDecoder Summary", "=" * 50, ""]
        lines.append(f"Categorical features ({len(self.cat_features)}):")
        for feat_name in self.cat_features:
            vocab = self.label_to_idx[feat_name]
            labels = list(vocab.keys())
            lines.append(f"  {feat_name}: {len(vocab)} classes")
            if len(labels) <= 20:
                for label, idx in sorted(vocab.items(), key=lambda x: x[1]):
                    lines.append(f"    {idx}: {label}")
            else:
                for label, idx in sorted(vocab.items(), key=lambda x: x[1])[:10]:
                    lines.append(f"    {idx}: {label}")
                lines.append(f"    ... ({len(vocab) - 10} more)")

        lines.append("")
        lines.append(f"Numerical features ({len(self.num_features)}):")
        for feat_name in self.num_features:
            scaler = self.encoder_decoder.continuous_encoders[feat_name]
            if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                lines.append(f"  {feat_name}: mean={mean:.4f}, scale={scale:.4f}")
            else:
                lines.append(f"  {feat_name}: (custom scaler)")
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"TensorDecoder(cat_features={self.cat_features}, "
            f"num_features={self.num_features})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_cat_feature(self, feature: Union[str, int]) -> str:
        if isinstance(feature, int):
            return self.cat_features[feature]
        if feature not in self.label_to_idx:
            raise KeyError(
                f"Unknown categorical feature '{feature}'. "
                f"Available: {self.cat_features}"
            )
        return feature

    def _resolve_num_feature(self, feature: Union[str, int]) -> str:
        if isinstance(feature, int):
            return self.num_features[feature]
        if feature not in [n for n in self.num_features]:
            raise KeyError(
                f"Unknown numerical feature '{feature}'. "
                f"Available: {self.num_features}"
            )
        return feature
