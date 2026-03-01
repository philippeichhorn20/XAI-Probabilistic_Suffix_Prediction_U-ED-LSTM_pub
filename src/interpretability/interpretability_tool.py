"""
Main InterpretabilityTool class for the suffix prediction model.

This class provides a high-level interface for computing and visualizing
feature attributions using multiple methods:
- Integrated Gradients (gradient-based)
- SHAP (Shapley values, perturbation-based)
- ICE (Individual Conditional Expectations, sensitivity analysis)

Usage:
    from interpretability import InterpretabilityTool

    # Initialize
    tool = InterpretabilityTool(model, data_set_categories)

    # Compute attribution map using different methods
    attr_map = tool.compute_attribution_map(
        process=process_data,           # Full process from dataset
        prefix_length=10,               # First 10 events are prefix
        target='Activity',              # Explain activity prediction
        target_class='auto',            # Explain predicted class (or specific index)
        suffix_scope='next',            # Explain just the next event prediction
        method='integrated_gradients'   # or 'shap' or 'ice'
    )

    # Output in different formats
    print(attr_map.to_string())         # Text representation
    tensor = attr_map.to_tensor()       # Raw tensor
    fig = attr_map.plot()               # Matplotlib figure with colored heatmap
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
import warnings
import numpy as np

from .attribution.integrated_gradients import IntegratedGradients
from .model.target_selectors import (
    TargetSelector,
    CategoricalTargetSelector,
    NumericalTargetSelector,
    create_target_selector
)
from .model.baselines import BaselineGenerator, ZeroBaseline, create_baseline_generator
from .model.model_wrapper import IGModelWrapper
from .visualization.attribution_vis import AttributionVisualizer
from .attribution.shap_explainer import SequenceSHAP
from .attribution.ice_explainer import SequenceICE


class AttributionMap:
    """
    Unified representation of attribution scores.

    Stores a 2D matrix where:
    - Y-axis (rows): Input features
    - X-axis (columns): Prefix steps

    Provides three output formats:
    - to_tensor(): Raw tensor for computation
    - to_string(): Printable text representation
    - plot(): Matplotlib figure with colored heatmap
    """

    def __init__(
        self,
        attributions: Dict[str, Tensor],
        feature_names: List[str],
        step_labels: List[str],
        target_description: str,
        convergence_delta: Optional[float] = None,
        prefix_length: int = 0,
        suffix_scope: str = 'next'
    ):
        self.attributions = attributions
        self.feature_names = feature_names
        self.step_labels = step_labels
        self.target_description = target_description
        self.convergence_delta = convergence_delta
        self.prefix_length = prefix_length
        self.suffix_scope = suffix_scope
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> np.ndarray:
        """Build 2D numpy array from attributions dict."""
        n_features = len(self.feature_names)
        n_steps = len(self.step_labels)

        matrix = np.zeros((n_features, n_steps))
        for i, name in enumerate(self.feature_names):
            if name in self.attributions:
                attr = self.attributions[name]
                if isinstance(attr, Tensor):
                    attr = attr.detach().cpu().numpy()
                attr = np.array(attr).flatten()
                matrix[i, :len(attr)] = attr[:n_steps]

        return matrix

    def to_tensor(self) -> Tensor:
        """Get attribution scores as a PyTorch tensor."""
        return torch.tensor(self._matrix, dtype=torch.float32)

    def to_numpy(self) -> np.ndarray:
        """Get attribution scores as a NumPy array."""
        return self._matrix.copy()

    def to_string(self, max_label_width: int = 20, value_format: str = '.4f') -> str:
        """Get a printable string representation."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"ATTRIBUTION MAP: {self.target_description}")
        lines.append(f"Prefix length: {self.prefix_length} | Suffix scope: {self.suffix_scope}")
        if self.convergence_delta is not None:
            lines.append(f"Convergence delta: {self.convergence_delta:.6f}")
        lines.append("=" * 80)
        lines.append("")

        def truncate(s, width):
            s = str(s)
            return s[:width-2] + ".." if len(s) > width else s

        step_labels = [truncate(l, 12) for l in self.step_labels]
        feature_labels = [truncate(f, max_label_width) for f in self.feature_names]

        header = " " * (max_label_width + 2)
        for label in step_labels:
            header += f"{label:>12}"
        lines.append(header)
        lines.append("-" * len(header))

        for i, feat_name in enumerate(feature_labels):
            row = f"{feat_name:>{max_label_width}} |"
            for j in range(len(step_labels)):
                val = self._matrix[i, j]
                row += f"{val:>12{value_format}}"
            lines.append(row)

        lines.append("-" * len(header))
        total_row = f"{'TOTAL':>{max_label_width}} |"
        for j in range(len(step_labels)):
            col_total = self._matrix[:, j].sum()
            total_row += f"{col_total:>12{value_format}}"
        lines.append(total_row)
        lines.append("=" * 80)

        return "\n".join(lines)

    def plot(
        self,
        figsize: Tuple[int, int] = (14, 8),
        cmap: str = 'RdBu_r',
        show_values: bool = True,
        title: Optional[str] = None,
        ax=None
    ):
        """Create a colored heatmap visualization."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        ax.grid(False)
        vmax = np.abs(self._matrix).max()
        vmin = -vmax if vmax > 0 else -1

        im = ax.imshow(self._matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', interpolation='nearest')

        step_labels = [l[:12] + ".." if len(str(l)) > 12 else str(l) for l in self.step_labels]
        feature_labels = [f[:18] + ".." if len(str(f)) > 18 else str(f) for f in self.feature_names]

        ax.set_xticks(range(len(step_labels)))
        ax.set_xticklabels(step_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(feature_labels)))
        ax.set_yticklabels(feature_labels)

        ax.set_xlabel('Prefix Step')
        ax.set_ylabel('Input Feature')
        ax.set_title(title or f"Attribution Map: {self.target_description}")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attribution Score')

        if show_values and self._matrix.shape[0] * self._matrix.shape[1] <= 100:
            for i in range(len(self.feature_names)):
                for j in range(len(self.step_labels)):
                    val = self._matrix[i, j]
                    color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                           color=color, fontsize=11)

        plt.tight_layout()
        return fig, ax

    def __repr__(self) -> str:
        delta_str = f"{self.convergence_delta:.6f}" if self.convergence_delta else "N/A"
        return (
            f"AttributionMap(\n"
            f"  target: {self.target_description}\n"
            f"  shape: {self._matrix.shape} (features x steps)\n"
            f"  prefix_length: {self.prefix_length}\n"
            f"  convergence_delta: {delta_str}\n"
            f")"
        )


class AttributionResult:
    """Container for attribution results with metadata."""

    def __init__(
        self,
        attributions: Tensor,
        target_description: str,
        baseline_description: str,
        convergence_delta: Optional[float] = None,
        feature_attributions: Optional[Dict[str, Tensor]] = None,
        resolved_target: Optional[Union[int, str]] = None,
        suffix_step: int = 0,
        n_steps: int = 50
    ):
        self.attributions = attributions
        self.target_description = target_description
        self.baseline_description = baseline_description
        self.convergence_delta = convergence_delta
        self.feature_attributions = feature_attributions
        self.resolved_target = resolved_target
        self.suffix_step = suffix_step
        self.n_steps = n_steps

    def __repr__(self) -> str:
        delta_str = f"{self.convergence_delta:.6f}" if self.convergence_delta else "N/A"
        return (
            f"AttributionResult(\n"
            f"  target: {self.target_description}\n"
            f"  baseline: {self.baseline_description}\n"
            f"  suffix_step: {self.suffix_step}\n"
            f"  convergence_delta: {delta_str}\n"
            f"  shape: {self.attributions.shape}\n"
            f")"
        )

    def to_numpy(self) -> 'np.ndarray':
        return self.attributions.detach().cpu().numpy()


class InterpretabilityTool:
    """
    High-level interface for computing and visualizing model attributions.

    This tool wraps the Integrated Gradients implementation and provides:
    - Easy specification of targets (activity, resource, time, etc.)
    - Support for 'auto' target (explain predicted class)
    - Multiple baseline strategies
    - Built-in visualization
    - Feature-level attribution aggregation
    """

    def __init__(
        self,
        model,
        data_set_categories: Optional[list] = None,
        device: Optional[str] = None
    ):
        self.model = model
        self.data_set_categories = data_set_categories or model.data_set_categories
        self.device = device or next(model.parameters()).device
        self.model_wrapper = None
        self.visualizer = None
        self._baseline_cache = {}
        self._feature_info = self._extract_feature_info()

    def _extract_feature_info(self) -> Dict[str, Dict]:
        """Extract feature information from model."""
        cat_categories, num_categories = self.data_set_categories
        feature_info = {'categorical': {}, 'numerical': {}}

        for name, size, label_dict in cat_categories:
            feature_info['categorical'][name] = {
                'size': size, 'labels': label_dict, 'type': 'categorical'
            }
        for name, size, info in num_categories:
            feature_info['numerical'][name] = {
                'size': size, 'info': info, 'type': 'numerical'
            }
        return feature_info

    def get_available_targets(self) -> Dict[str, List[str]]:
        """Get list of available target outputs."""
        return {
            'categorical': list(self._feature_info['categorical'].keys()),
            'numerical': list(self._feature_info['numerical'].keys())
        }

    def get_feature_type(self, feature_name: str) -> str:
        """Determine if a feature is categorical or numerical."""
        if feature_name in self._feature_info['categorical']:
            return 'categorical'
        elif feature_name in self._feature_info['numerical']:
            return 'numerical'
        else:
            raise ValueError(f"Unknown feature: {feature_name}")

    def _get_or_create_wrapper(self, suffix_step: int) -> IGModelWrapper:
        """Get or create model wrapper for the given suffix step."""
        if self.model_wrapper is None or self.model_wrapper.suffix_step != suffix_step:
            self.model_wrapper = IGModelWrapper(self.model, suffix_step=suffix_step)
        return self.model_wrapper

    def _get_baseline_generator(self, strategy: str) -> BaselineGenerator:
        """Get or create baseline generator."""
        if strategy in self._baseline_cache:
            return self._baseline_cache[strategy]

        if strategy == 'zero':
            gen = ZeroBaseline()
        elif strategy == 'padding':
            gen = create_baseline_generator('padding', embeddings=self.model.embeddings_enc)
        elif strategy == 'uniform':
            gen = create_baseline_generator('uniform', embeddings=self.model.embeddings_enc)
        else:
            raise ValueError(f"Unknown baseline strategy: {strategy}")

        self._baseline_cache[strategy] = gen
        return gen

    def _to_device(self, data: Tuple[List[Tensor], List[Tensor]]) -> Tuple[List[Tensor], List[Tensor]]:
        """Move data to the correct device."""
        cat_tensors, num_tensors = data
        return ([t.to(self.device) for t in cat_tensors],
                [t.to(self.device) for t in num_tensors])

    # =========================================================================
    # Core IG computation
    # =========================================================================

    def compute_integrated_gradients(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        target_output: str,
        target_value: Union[int, str] = 'auto',
        suffix_step: int = 0,
        n_steps: int = 50,
        baseline: Union[str, BaselineGenerator] = 'zero',
        return_convergence_delta: bool = True,
        aggregate_by_feature: bool = True
    ) -> AttributionResult:
        """
        Compute Integrated Gradients attributions for encoder prefix and decoder SOS jointly.

        Uses a single joint IG computation over both inputs simultaneously,
        which satisfies the completeness axiom.

        Args:
            prefix: Input prefix sequence as (categorical_tensors, numerical_tensors)
            target_output: Name of output to explain
            target_value: For categorical: class index or 'auto'. For numerical: 'mean' or 'var'
            suffix_step: Which suffix step to explain (0-indexed)
            n_steps: Number of integration steps
            baseline: Baseline strategy ('zero', 'padding', 'uniform') or BaselineGenerator
            return_convergence_delta: Whether to compute completeness check
            aggregate_by_feature: Whether to compute per-feature attributions

        Returns:
            AttributionResult with attributions and metadata
        """
        prefix = self._to_device(prefix)
        wrapper = self._get_or_create_wrapper(suffix_step)
        feature_type = self.get_feature_type(target_output)

        # Create target selector
        target_selector = create_target_selector(
            target_output=target_output,
            target_value=target_value,
            feature_type=feature_type
        )

        # Create baseline generator
        baseline_gen = baseline if isinstance(baseline, BaselineGenerator) else self._get_baseline_generator(baseline)

        # Embed inputs
        embedded_prefix = wrapper.embed_prefix(prefix)
        embedded_sos = wrapper.embed_sos(prefix)

        # Generate baselines
        baseline_prefix = baseline_gen.generate(prefix, embedded_prefix)
        baseline_sos = torch.zeros_like(embedded_sos)

        # Pre-resolve 'auto' target
        if target_value == 'auto' and feature_type == 'categorical':
            with torch.no_grad():
                preds = wrapper.forward(embedded_prefix, embedded_sos)
                target_selector.select(preds, suffix_step)

        # Joint IG over both inputs simultaneously
        emb_prefix_grad = embedded_prefix.clone().requires_grad_(True)
        emb_sos_grad = embedded_sos.clone().requires_grad_(True)
        forward_fn = wrapper.combined_forward_fn(target_selector)
        ig = IntegratedGradients(forward_fn, multiply_by_inputs=True)

        if return_convergence_delta:
            (enc_attrs, sos_attrs), delta = ig.compute(
                inputs=(emb_prefix_grad, emb_sos_grad),
                baselines=(baseline_prefix, baseline_sos),
                n_steps=n_steps,
                return_convergence_delta=True
            )
        else:
            enc_attrs, sos_attrs = ig.compute(
                inputs=(emb_prefix_grad, emb_sos_grad),
                baselines=(baseline_prefix, baseline_sos),
                n_steps=n_steps,
                return_convergence_delta=False
            )
            delta = None

        attr_tensor = enc_attrs

        # Aggregate by feature
        feature_attrs = None
        if aggregate_by_feature:
            if self.visualizer is None:
                self.visualizer = AttributionVisualizer(
                    feature_names=wrapper.get_encoder_feature_info(),
                    data_set_categories=self.data_set_categories
                )
            feature_attrs = self.visualizer.aggregate_by_feature(attr_tensor)

        resolved = target_selector.resolved_class if isinstance(target_selector, CategoricalTargetSelector) else None

        return AttributionResult(
            attributions=attr_tensor,
            target_description=target_selector.get_description(),
            baseline_description=baseline_gen.get_description(),
            convergence_delta=delta,
            feature_attributions=feature_attrs,
            resolved_target=resolved,
            suffix_step=suffix_step,
            n_steps=n_steps
        )

    # =========================================================================
    # Full chain attribution (encoder + decoder chain)
    # =========================================================================

    def compute_full_chain_attribution(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        target_output: str,
        target_value: Union[int, str] = 'auto',
        suffix_step: int = 0,
        n_steps: int = 50,
        baseline: str = 'zero',
        actual_prefix_length: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Compute attributions for both encoder prefix AND decoder input chain.

        For predicting suffix step t:
        - Encoder: attributions for enc(prefix[0]), ..., enc(prefix[n-1])
        - Decoder: attributions for dec(SOS), dec(pred_0), ..., dec(pred_{t-1})

        Args:
            actual_prefix_length: Number of prefix events to include (uses last N).
                                 If None, uses full embedded prefix length.

        Returns:
            Dict with:
            - encoder_attributions: {feature_name: [attr_per_position]}
            - decoder_attributions: {feature_name: [attr_per_position]}
            - encoder_labels: labels for encoder positions
            - decoder_labels: labels for decoder positions (SOS, pred_0, ...)
        """
        prefix = self._to_device(prefix)
        wrapper = self._get_or_create_wrapper(suffix_step)
        feature_type = self.get_feature_type(target_output)

        target_selector = create_target_selector(
            target_output=target_output,
            target_value=target_value,
            feature_type=feature_type
        )
        baseline_gen = self._get_baseline_generator(baseline)

        # Embed prefix and SOS
        embedded_prefix = wrapper.embed_prefix(prefix)
        embedded_sos = wrapper.embed_sos(prefix)

        # Pre-resolve 'auto' target
        if target_value == 'auto' and feature_type == 'categorical':
            with torch.no_grad():
                preds = wrapper.forward(embedded_prefix, embedded_sos)
                target_selector.select(preds, suffix_step)

        # === Joint IG over encoder prefix and SOS ===
        baseline_prefix = baseline_gen.generate(prefix, embedded_prefix)
        baseline_sos = torch.zeros_like(embedded_sos)
        emb_prefix_grad = embedded_prefix.clone().requires_grad_(True)
        emb_sos_grad = embedded_sos.clone().requires_grad_(True)
        forward_fn = wrapper.combined_forward_fn(target_selector)
        ig = IntegratedGradients(forward_fn, multiply_by_inputs=True)
        (enc_attrs, sos_attrs), delta = ig.compute(
            inputs=(emb_prefix_grad, emb_sos_grad),
            baselines=(baseline_prefix, baseline_sos),
            n_steps=n_steps, return_convergence_delta=True
        )

        enc_attr_tensor = enc_attrs   # [prefix_len, batch, enc_dim]

        if suffix_step > 0:
            # For suffix_step > 0, also attribute to intermediate decoder predictions.
            # The SOS attribution is already captured in sos_attrs above.
            # Here we get attributions for [pred_0, ..., pred_{t-1}].
            decoder_inputs = wrapper.get_decoder_inputs_for_step(
                embedded_prefix, embedded_sos, suffix_step
            )  # [suffix_step + 1, batch, dec_dim]
            # Take only intermediate predictions (skip SOS at index 0)
            intermediate_inputs = decoder_inputs[1:]  # [suffix_step, batch, dec_dim]

            if intermediate_inputs.shape[0] > 0:
                intermediate_grad = intermediate_inputs.clone().requires_grad_(True)
                baseline_intermediate = torch.zeros_like(intermediate_inputs)
                forward_fn_dec = wrapper.decoder_chain_forward_fn(target_selector, embedded_prefix)

                # Build decoder chain with SOS fixed + variable intermediates
                fixed_sos_for_chain = embedded_sos.detach()
                def chain_forward(intermediates):
                    full_chain = torch.cat([fixed_sos_for_chain, intermediates], dim=0)
                    preds = wrapper.forward_from_decoder_sequence(
                        embedded_prefix.detach(), full_chain, suffix_step
                    )
                    return target_selector.select_direct(preds)

                ig_chain = IntegratedGradients(chain_forward, multiply_by_inputs=True)
                chain_attrs_tuple = ig_chain.compute(
                    inputs=(intermediate_grad,), baselines=(baseline_intermediate,),
                    n_steps=n_steps, return_convergence_delta=False
                )
                chain_attrs = chain_attrs_tuple[0]  # [suffix_step, batch, dec_dim]
                # Concatenate SOS attrs + intermediate attrs for full decoder tensor
                dec_attr_tensor = torch.cat([sos_attrs, chain_attrs], dim=0)
            else:
                dec_attr_tensor = sos_attrs  # [1, batch, dec_dim]
        else:
            # suffix_step == 0: decoder input is just SOS
            dec_attr_tensor = sos_attrs  # [1, batch, dec_dim]

        # Determine actual prefix length to use
        prefix_len_to_use = actual_prefix_length if actual_prefix_length else enc_attr_tensor.shape[0]

        # Aggregate by feature
        if self.visualizer is None:
            self.visualizer = AttributionVisualizer(
                feature_names=wrapper.get_encoder_feature_info(),
                data_set_categories=self.data_set_categories
            )
        decoder_visualizer = AttributionVisualizer(
            feature_names=wrapper.get_decoder_feature_info(),
            data_set_categories=self.data_set_categories
        )

        enc_feature_attrs = self.visualizer.aggregate_by_feature(enc_attr_tensor)
        dec_feature_attrs = decoder_visualizer.aggregate_by_feature(dec_attr_tensor)

        # Get prediction names for decoder labels
        prediction_names = self._get_prediction_names(
            wrapper, embedded_prefix, embedded_sos, suffix_step, target_output
        )

        # Generate labels (only for the actual prefix positions we'll use)
        encoder_labels = self._get_prefix_labels_from_tensor(prefix, prefix_len_to_use)
        decoder_labels = self._get_decoder_chain_labels(suffix_step, prediction_names)

        return {
            'encoder_attributions': enc_feature_attrs,
            'decoder_attributions': dec_feature_attrs,
            'encoder_labels': encoder_labels,
            'decoder_labels': decoder_labels,
            'convergence_delta': delta,
            'target_description': target_selector.get_description(),
            'suffix_step': suffix_step
        }

    def _get_prefix_labels_from_tensor(self, prefix, length) -> List[str]:
        """Get labels for prefix positions."""
        cat_tensors, _ = prefix
        if not cat_tensors:
            return [f"enc({i})" for i in range(length)]

        cat_categories, _ = self.data_set_categories
        if cat_categories:
            label_map = {v: k for k, v in cat_categories[0][2].items()}
            indices = cat_tensors[0][0].detach().cpu().numpy()
            start_idx = len(indices) - length
            indices = indices[start_idx:]
            return [f"enc({label_map.get(int(idx), str(idx))})" for idx in indices]

        return [f"enc({i})" for i in range(length)]

    def _get_prediction_names(
        self,
        wrapper,
        embedded_prefix: Tensor,
        embedded_sos: Tensor,
        suffix_step: int,
        target_output: str
    ) -> List[str]:
        """Get the predicted activity names for each suffix step."""
        if suffix_step == 0:
            return []

        # Run forward pass to get all predictions
        with torch.no_grad():
            cat_preds, num_preds = wrapper.forward(embedded_prefix, embedded_sos)

        # Get label map for the target output
        label_map = {}
        cat_categories, _ = self.data_set_categories
        for name, _, labels in cat_categories:
            if name == target_output:
                label_map = {v: k for k, v in labels.items()}
                break

        # Extract prediction names for steps 0 to suffix_step-1
        prediction_names = []
        key = f"{target_output}_mean"
        if key in cat_preds:
            logits = cat_preds[key]  # [suffix_len, batch, num_classes]
            for t in range(suffix_step):
                pred_class = logits[t].argmax(dim=-1).item()
                pred_name = label_map.get(pred_class, str(pred_class))
                # Truncate long names
                if len(pred_name) > 15:
                    pred_name = pred_name[:12] + "..."
                prediction_names.append(pred_name)

        return prediction_names

    def _get_decoder_chain_labels(
        self,
        suffix_step: int,
        prediction_names: Optional[List[str]] = None
    ) -> List[str]:
        """Get labels for decoder chain positions with actual prediction names."""
        labels = ["dec(SOS)"]
        for t in range(suffix_step):
            if prediction_names and t < len(prediction_names):
                labels.append(f"dec({prediction_names[t]})")
            else:
                labels.append(f"dec(pred_{t})")
        return labels

    # =========================================================================
    # Encoder vs Decoder split (legacy, SOS only)
    # =========================================================================

    def compute_encoder_decoder_attribution_split(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        target_output: str,
        target_value: Union[int, str] = 'auto',
        suffix_step: int = 0,
        n_steps: int = 50,
        baseline: str = 'zero'
    ) -> Dict[str, any]:
        """
        Compute how much attribution comes from encoder prefix vs decoder SOS.

        The encoder-decoder architecture has two input paths:
        1. Encoder prefix → hidden states
        2. Decoder SOS (last prefix event) → predictions

        Uses a single joint IG computation over both inputs simultaneously,
        which satisfies the completeness axiom (sum of all attributions =
        F(input) - F(baseline)).

        Returns:
            Dict with encoder/decoder totals, fractions, and per-feature attributions
        """
        prefix = self._to_device(prefix)
        wrapper = self._get_or_create_wrapper(suffix_step)
        feature_type = self.get_feature_type(target_output)

        target_selector = create_target_selector(
            target_output=target_output,
            target_value=target_value,
            feature_type=feature_type
        )
        baseline_gen = self._get_baseline_generator(baseline)

        # Embed both inputs
        embedded_prefix = wrapper.embed_prefix(prefix)
        embedded_sos = wrapper.embed_sos(prefix)

        # Generate baselines
        baseline_prefix = baseline_gen.generate(prefix, embedded_prefix)
        baseline_sos = torch.zeros_like(embedded_sos)

        # Pre-resolve 'auto' target
        if target_value == 'auto' and feature_type == 'categorical':
            with torch.no_grad():
                preds = wrapper.forward(embedded_prefix, embedded_sos)
                target_selector.select(preds, suffix_step)

        # Joint IG over both inputs simultaneously
        emb_prefix_grad = embedded_prefix.clone().requires_grad_(True)
        emb_sos_grad = embedded_sos.clone().requires_grad_(True)
        forward_fn = wrapper.combined_forward_fn(target_selector)
        ig = IntegratedGradients(forward_fn, multiply_by_inputs=True)
        (enc_attrs, sos_attrs), delta = ig.compute(
            inputs=(emb_prefix_grad, emb_sos_grad),
            baselines=(baseline_prefix, baseline_sos),
            n_steps=n_steps, return_convergence_delta=True
        )

        enc_attr_tensor = enc_attrs
        sos_attr_tensor = sos_attrs

        # Compute totals and fractions
        enc_total = enc_attr_tensor.abs().sum().item()
        sos_total = sos_attr_tensor.abs().sum().item()
        total = enc_total + sos_total
        enc_fraction = enc_total / total if total > 0 else 0.0
        sos_fraction = sos_total / total if total > 0 else 0.0

        # Aggregate by feature
        if self.visualizer is None:
            self.visualizer = AttributionVisualizer(
                feature_names=wrapper.get_encoder_feature_info(),
                data_set_categories=self.data_set_categories
            )
        decoder_visualizer = AttributionVisualizer(
            feature_names=wrapper.get_decoder_feature_info(),
            data_set_categories=self.data_set_categories
        )

        enc_feature_attrs = self.visualizer.aggregate_by_feature(enc_attr_tensor)
        sos_feature_attrs = decoder_visualizer.aggregate_by_feature(sos_attr_tensor)

        return {
            'encoder_total': enc_total,
            'decoder_sos_total': sos_total,
            'encoder_fraction': enc_fraction,
            'decoder_sos_fraction': sos_fraction,
            'encoder_attributions': enc_feature_attrs,
            'decoder_sos_attributions': sos_feature_attrs,
            'encoder_feature_totals': {n: a.abs().sum().item() for n, a in enc_feature_attrs.items()},
            'decoder_sos_feature_totals': {n: a.abs().sum().item() for n, a in sos_feature_attrs.items()},
            'convergence_delta': delta,
            'target_description': target_selector.get_description()
        }

    @staticmethod
    def format_attribution_split(split_result: Dict) -> str:
        """Format the encoder/decoder attribution split as a readable string."""
        lines = []
        lines.append("=" * 70)
        lines.append("ENCODER vs DECODER (SOS) ATTRIBUTION SPLIT")
        lines.append(f"Target: {split_result['target_description']}")
        lines.append("=" * 70)
        lines.append("")

        enc_pct = split_result['encoder_fraction'] * 100
        dec_pct = split_result['decoder_sos_fraction'] * 100

        lines.append(f"  Encoder (full prefix):    {enc_pct:6.2f}%  (total: {split_result['encoder_total']:.4f})")
        lines.append(f"  Decoder (last event/SOS): {dec_pct:6.2f}%  (total: {split_result['decoder_sos_total']:.4f})")
        lines.append("")

        bar_len = 50
        enc_bar = int(enc_pct / 100 * bar_len)
        dec_bar = bar_len - enc_bar
        lines.append(f"  [{'█' * enc_bar}{'░' * dec_bar}]")
        lines.append(f"   {'Encoder':<25} {'Decoder SOS':>25}")
        lines.append("")

        lines.append("-" * 70)
        lines.append("Per-feature breakdown:")
        lines.append(f"  {'Feature':<25} {'Encoder':>12} {'Decoder SOS':>12} {'Enc %':>8}")
        lines.append("-" * 70)

        enc_totals = split_result['encoder_feature_totals']
        dec_totals = split_result['decoder_sos_feature_totals']

        for feat in enc_totals.keys():
            enc_val = enc_totals.get(feat, 0)
            dec_val = dec_totals.get(feat, 0)
            feat_total = enc_val + dec_val
            enc_feat_pct = (enc_val / feat_total * 100) if feat_total > 0 else 0
            lines.append(f"  {feat:<25} {enc_val:>12.4f} {dec_val:>12.4f} {enc_feat_pct:>7.1f}%")

        lines.append("=" * 70)
        return "\n".join(lines)

    # =========================================================================
    # High-level attribution map API
    # =========================================================================

    def extract_prefix_from_process(
        self,
        process: Tuple[List[Tensor], List[Tensor]],
        prefix_length: int,
        device: Optional[str] = None,
        skip_padding: bool = True
    ) -> Tuple[int, Tuple[List[Tensor], List[Tensor]]]:
        """Extract a prefix from a full process sequence."""
        device = device or self.device
        cat_case, num_case = process

        process_seq_len = cat_case[0].shape[0] if cat_case else num_case[0].shape[0]
        model_seq_len = process_seq_len

        start_idx = 0
        if skip_padding and cat_case:
            first_cat = cat_case[0].numpy() if hasattr(cat_case[0], 'numpy') else cat_case[0]
            for i, v in enumerate(first_cat):
                if v > 0:
                    start_idx = i
                    break

        current_prefix = (
            [torch.zeros(1, model_seq_len, dtype=t.dtype) for t in cat_case],
            [torch.zeros(1, model_seq_len, dtype=t.dtype) for t in num_case],
        )

        max_events = process_seq_len - start_idx
        actual_length = min(prefix_length, max_events)

        for i in range(actual_length):
            src_idx = start_idx + i
            for j in range(len(current_prefix[0])):
                current_prefix[0][j][0] = torch.roll(current_prefix[0][j][0], shifts=-1)
                current_prefix[0][j][0, -1] = cat_case[j][src_idx]
            for j in range(len(current_prefix[1])):
                current_prefix[1][j][0] = torch.roll(current_prefix[1][j][0], shifts=-1)
                current_prefix[1][j][0, -1] = num_case[j][src_idx]

        current_prefix = (
            [t.to(device) for t in current_prefix[0]],
            [t.to(device) for t in current_prefix[1]],
        )
        return actual_length, current_prefix

    def compute_attribution_map(
        self,
        process: Tuple[List[Tensor], List[Tensor]],
        prefix_length: int,
        target: str,
        target_class: Union[int, str] = 'auto',
        suffix_scope: str = 'next',
        suffix_step: int = 0,
        method: str = 'integrated_gradients',
        n_steps: int = 50,
        baseline: str = 'zero',
        n_samples: int = 100
    ) -> AttributionMap:
        """
        Compute an attribution map for a process.

        Args:
            process: Full process data as (cat_tensors, num_tensors)
            prefix_length: Number of events to use as prefix
            target: Name of the output to explain
            target_class: 'auto' for predicted class, or specific class index
            suffix_scope: 'next' (step 0) or 'step' (use suffix_step param)
            suffix_step: Which suffix step to explain (0-indexed)
            method: 'integrated_gradients', 'shap', or 'ice'
            n_steps: Number of integration steps for IG
            baseline: Baseline strategy
            n_samples: Number of samples for SHAP

        Returns:
            AttributionMap object
        """
        actual_prefix_length, prefix = self.extract_prefix_from_process(process, prefix_length)

        feature_type = self.get_feature_type(target)
        target_value = 'mean' if feature_type == 'numerical' else target_class

        if suffix_scope == 'next':
            suffix_step = 0

        feature_names = self._get_ordered_feature_names()

        # Compute attributions using the specified method
        if method == 'integrated_gradients':
            # IG returns full chain: encoder positions + decoder chain positions
            filtered_attributions, convergence_delta, resolved_target, step_labels = \
                self._compute_ig_attributions(prefix, target, target_value, suffix_step,
                                             n_steps, baseline, actual_prefix_length)
        elif method == 'shap':
            step_labels = self._get_prefix_step_labels(prefix, actual_prefix_length)
            filtered_attributions, convergence_delta, resolved_target = \
                self._compute_shap_attributions(prefix, target, target_value, suffix_step,
                                               n_samples, baseline, actual_prefix_length)
        elif method == 'ice':
            step_labels = self._get_prefix_step_labels(prefix, actual_prefix_length)
            filtered_attributions, convergence_delta, resolved_target = \
                self._compute_ice_attributions(prefix, target, target_value, suffix_step,
                                              actual_prefix_length)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Build target description
        if feature_type == 'categorical' and resolved_target is not None:
            label_map = self._feature_info['categorical'].get(target, {}).get('labels', {})
            label_map_inv = {v: k for k, v in label_map.items()}
            class_name = label_map_inv.get(resolved_target, str(resolved_target))
            target_desc = f"{target} = {class_name} (class {resolved_target}) [{method}]"
        else:
            target_desc = f"{target} prediction [{method}]"

        return AttributionMap(
            attributions=filtered_attributions,
            feature_names=feature_names,
            step_labels=step_labels,
            target_description=target_desc,
            convergence_delta=convergence_delta,
            prefix_length=actual_prefix_length,
            suffix_scope=suffix_scope
        )

    def _get_prefix_step_labels(self, prefix, actual_prefix_length) -> List[str]:
        """Get human-readable labels for each prefix step."""
        cat_tensors, _ = prefix
        if not cat_tensors:
            return [f"Step {i}" for i in range(actual_prefix_length)]

        cat_categories, _ = self.data_set_categories
        if cat_categories:
            label_map = {v: k for k, v in cat_categories[0][2].items()}
            indices = cat_tensors[0][0].detach().cpu().numpy()
            start_idx = len(indices) - actual_prefix_length
            indices = indices[start_idx:]
            return [label_map.get(int(idx), str(idx)) for idx in indices]

        return [f"Step {i}" for i in range(actual_prefix_length)]

    def _get_ordered_feature_names(self) -> List[str]:
        """Get all feature names in a consistent order."""
        names = []
        cat_categories, num_categories = self.data_set_categories
        for name, _, _ in cat_categories:
            names.append(name)
        for name, _, _ in num_categories:
            names.append(name)
        return names

    def _compute_ig_attributions(self, prefix, target, target_value, suffix_step,
                                 n_steps, baseline, actual_prefix_length):
        """Compute attributions using Integrated Gradients.

        Computes both encoder (prefix) and decoder chain (SOS + intermediate predictions).

        For suffix step t:
        - Encoder positions: enc(a), enc(b), enc(c) - one per prefix event
        - Decoder positions: dec(SOS), dec(pred_0), ..., dec(pred_{t-1})

        Returns attributions concatenated: [encoder positions | decoder positions]
        Also returns step_labels for the combined positions.
        """
        # Compute full chain attributions
        chain_result = self.compute_full_chain_attribution(
            prefix=prefix, target_output=target, target_value=target_value,
            suffix_step=suffix_step, n_steps=n_steps, baseline=baseline,
            actual_prefix_length=actual_prefix_length
        )

        # Combine encoder and decoder attributions into single array per feature
        filtered_attributions = {}
        enc_attrs = chain_result['encoder_attributions']
        dec_attrs = chain_result['decoder_attributions']

        for name in enc_attrs.keys():
            # Get encoder attributions for this feature
            enc_attr = enc_attrs[name]
            enc_np = enc_attr.detach().cpu().numpy().flatten()
            # Take exactly the last actual_prefix_length values
            enc_np = enc_np[-actual_prefix_length:]

            # Get decoder attributions for this feature (may have different name in decoder)
            dec_np = np.zeros(suffix_step + 1)  # SOS + suffix_step intermediate predictions
            if name in dec_attrs:
                dec_attr = dec_attrs[name]
                dec_raw = dec_attr.detach().cpu().numpy().flatten()
                # Decoder attrs shape: [suffix_step + 1] (one per decoder input)
                dec_np = dec_raw[:suffix_step + 1]

            # Concatenate: [encoder positions | decoder positions]
            filtered_attributions[name] = np.concatenate([enc_np, dec_np])

        # Build combined step labels
        step_labels = chain_result['encoder_labels'] + chain_result['decoder_labels']

        # Resolve target class from the result
        resolved_target = None
        target_desc = chain_result.get('target_description', '')
        if 'class' in target_desc:
            import re
            match = re.search(r'class (\d+)', target_desc)
            if match:
                resolved_target = int(match.group(1))

        convergence_delta = chain_result.get('convergence_delta', 0)

        return filtered_attributions, convergence_delta, resolved_target, step_labels

    def _compute_shap_attributions(self, prefix, target, target_value, suffix_step,
                                   n_samples, baseline, actual_prefix_length):
        """Compute attributions using SHAP."""
        shap_explainer = SequenceSHAP(
            model=self.model, data_set_categories=self.data_set_categories, device=self.device
        )

        shap_values = shap_explainer.compute(
            prefix=prefix, target_output=target, target_value=target_value,
            suffix_step=suffix_step, n_samples=n_samples, baseline=baseline, feature_level=False
        )

        filtered_attributions = {}
        for name, attr in shap_values.items():
            attr_np = attr.numpy() if hasattr(attr, 'numpy') else np.array(attr)
            # Always take exactly the last actual_prefix_length values
            # to ensure consistent shape across all suffix steps
            filtered_attributions[name] = attr_np[-actual_prefix_length:]

        resolved_target = self._resolve_target(prefix, target, target_value, suffix_step)
        return filtered_attributions, None, resolved_target

    def _compute_ice_attributions(self, prefix, target, target_value, suffix_step, actual_prefix_length):
        """Compute attributions using ICE sensitivity analysis."""
        ice_explainer = SequenceICE(
            model=self.model, data_set_categories=self.data_set_categories, device=self.device
        )

        sensitivities = ice_explainer.compute_sensitivity(
            prefix=prefix, target_output=target, target_value=target_value, suffix_step=suffix_step
        )

        filtered_attributions = {}
        for name, attr in sensitivities.items():
            attr_np = attr.numpy() if hasattr(attr, 'numpy') else np.array(attr)
            # Always take exactly the last actual_prefix_length values
            # to ensure consistent shape across all suffix steps
            filtered_attributions[name] = attr_np[-actual_prefix_length:]

        resolved_target = self._resolve_target(prefix, target, target_value, suffix_step)
        return filtered_attributions, None, resolved_target

    def _resolve_target(self, prefix, target, target_value, suffix_step):
        """Resolve 'auto' target to actual class index."""
        if target_value != 'auto':
            return None
        self.model.eval()
        with torch.no_grad():
            predictions, _, _, _ = self.model(prefix)
            cat_preds, _ = predictions
            key = f"{target}_mean"
            if key in cat_preds:
                logits = cat_preds[key]
                if logits.dim() == 3:
                    logits = logits[suffix_step]
                return logits.argmax(dim=-1).item()
        return None

    # =========================================================================
    # Prediction and visualization helpers
    # =========================================================================

    def get_prediction(self, prefix: Tuple[List[Tensor], List[Tensor]], suffix_step: int = 0) -> Dict[str, Dict]:
        """Get model prediction for a prefix."""
        prefix = self._to_device(prefix)
        self.model.eval()

        with torch.no_grad():
            predictions, _, _, _ = self.model(prefix)

        cat_preds, num_preds = predictions
        result = {'categorical': {}, 'numerical': {}}

        for key, logits in cat_preds.items():
            if '_mean' in key:
                feat_name = key.replace('_mean', '')
                if logits.dim() == 3:
                    logits = logits[suffix_step]
                probs = torch.softmax(logits, dim=-1)
                pred_idx = probs.argmax(dim=-1).item()
                pred_prob = probs[0, pred_idx].item()

                label_map = self._feature_info['categorical'].get(feat_name, {}).get('labels', {})
                label_map_inv = {v: k for k, v in label_map.items()}
                pred_name = label_map_inv.get(pred_idx, str(pred_idx))

                result['categorical'][feat_name] = {
                    'index': pred_idx, 'name': pred_name, 'probability': pred_prob
                }

        for key, value in num_preds.items():
            if '_mean' in key:
                feat_name = key.replace('_mean', '')
                if value.dim() == 3:
                    value = value[suffix_step]
                result['numerical'][feat_name] = {'value': value[0, 0].item()}

        return result

    def plot_attributions(self, result: AttributionResult, prefix, plot_type: str = 'heatmap', **kwargs):
        """Visualize attributions."""
        if self.visualizer is None:
            wrapper = self._get_or_create_wrapper(0)
            self.visualizer = AttributionVisualizer(
                feature_names=wrapper.get_encoder_feature_info(),
                data_set_categories=self.data_set_categories
            )

        if plot_type == 'heatmap':
            return self.visualizer.plot_heatmap(
                result.attributions, prefix, title=f"IG Attributions: {result.target_description}", **kwargs
            )
        elif plot_type == 'feature_importance':
            return self.visualizer.plot_feature_importance(
                result.attributions, title=f"Feature Importance: {result.target_description}", **kwargs
            )
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
