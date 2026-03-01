"""
Visualization utilities for Integrated Gradients attributions.

Provides various ways to visualize feature attributions:
- Heatmaps over sequences
- Bar charts for feature importance
- Interactive HTML visualizations
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class AttributionVisualizer:
    """
    Visualizer for Integrated Gradients attributions.

    Supports multiple visualization formats:
    - Matplotlib heatmaps
    - Text-based visualization (for terminal)
    - HTML visualization
    """

    def __init__(
        self,
        feature_names: Optional[Dict[str, List[Tuple[str, int, int]]]] = None,
        data_set_categories: Optional[list] = None
    ):
        """
        Initialize the visualizer.

        Args:
            feature_names: Dict with 'categorical' and 'numerical' keys,
                          containing (name, start_idx, end_idx) tuples
            data_set_categories: Dataset category information for label mapping
        """
        self.feature_names = feature_names
        self.data_set_categories = data_set_categories
        self._label_maps = self._build_label_maps() if data_set_categories else {}

    def _build_label_maps(self) -> Dict[str, Dict[int, str]]:
        """Build mapping from indices to labels for categorical features."""
        if not self.data_set_categories:
            return {}

        label_maps = {}
        cat_categories, _ = self.data_set_categories

        for name, size, label_dict in cat_categories:
            # Invert the dictionary: {label_name: index} -> {index: label_name}
            label_maps[name] = {v: k for k, v in label_dict.items()}

        return label_maps

    def aggregate_attributions(
        self,
        attributions: Tensor,
        aggregation: str = 'l2_norm'
    ) -> Tensor:
        """
        Aggregate attributions across embedding dimensions.

        Args:
            attributions: Attribution tensor [seq_len, batch, embed_dim]
            aggregation: 'l2_norm', 'sum', 'mean', 'abs_sum', 'abs_max'

        Returns:
            Aggregated tensor [seq_len, batch] or [seq_len] if batch=1
        """
        if aggregation == 'l2_norm':
            result = torch.norm(attributions, p=2, dim=-1)
        elif aggregation == 'sum':
            result = attributions.sum(dim=-1)
        elif aggregation == 'mean':
            result = attributions.mean(dim=-1)
        elif aggregation == 'abs_sum':
            result = attributions.abs().sum(dim=-1)
        elif aggregation == 'abs_max':
            result = attributions.abs().max(dim=-1)[0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Squeeze batch dimension if batch_size=1
        if result.dim() > 1 and result.shape[1] == 1:
            result = result.squeeze(1)

        return result

    def aggregate_by_feature(
        self,
        attributions: Tensor,
        aggregation: str = 'l2_norm'
    ) -> Dict[str, Tensor]:
        """
        Aggregate attributions by input feature.

        Args:
            attributions: Attribution tensor [seq_len, batch, embed_dim]
            aggregation: Aggregation method

        Returns:
            Dict mapping feature names to aggregated attributions [seq_len]
        """
        if self.feature_names is None:
            raise ValueError("Feature names required for per-feature aggregation")

        feature_attrs = {}

        # Process categorical features
        for name, start, end in self.feature_names.get('categorical', []):
            feat_attr = attributions[:, :, start:end]
            feature_attrs[name] = self.aggregate_attributions(feat_attr, aggregation)

        # Process numerical features
        for name, start, end in self.feature_names.get('numerical', []):
            feat_attr = attributions[:, :, start:end]
            feature_attrs[name] = self.aggregate_attributions(feat_attr, aggregation)

        return feature_attrs

    def plot_heatmap(
        self,
        attributions: Tensor,
        prefix: Tuple[List[Tensor], List[Tensor]],
        feature_name: Optional[str] = None,
        aggregation: str = 'l2_norm',
        figsize: Tuple[int, int] = (12, 6),
        cmap: str = 'RdBu_r',
        title: Optional[str] = None,
        show_values: bool = True,
        ax=None
    ):
        """
        Plot heatmap of attributions over the sequence.

        Args:
            attributions: Attribution tensor [seq_len, batch, embed_dim]
            prefix: Original prefix data for labeling
            feature_name: If specified, show only this feature's attributions
            aggregation: How to aggregate across embedding dims
            figsize: Figure size
            cmap: Colormap name
            title: Plot title
            show_values: Whether to show attribution values on heatmap
            ax: Matplotlib axes (creates new figure if None)

        Returns:
            Matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")

        # Aggregate attributions
        if feature_name and self.feature_names:
            feature_attrs = self.aggregate_by_feature(attributions, aggregation)
            if feature_name not in feature_attrs:
                raise ValueError(f"Feature '{feature_name}' not found")
            attr_values = feature_attrs[feature_name].detach().cpu().numpy()
            y_labels = [feature_name]
            attr_matrix = attr_values.reshape(1, -1)
        else:
            # Aggregate all features
            attr_values = self.aggregate_attributions(attributions, aggregation)
            attr_values = attr_values.detach().cpu().numpy()
            y_labels = ['All features']
            attr_matrix = attr_values.reshape(1, -1)

        # Get sequence labels from prefix
        x_labels = self._get_sequence_labels(prefix)

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Normalize for colormap (center at 0)
        vmax = max(abs(attr_matrix.min()), abs(attr_matrix.max()))
        vmin = -vmax if vmax > 0 else -1

        # Plot heatmap
        im = ax.imshow(attr_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attribution')

        # Set labels
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Feature')

        if title:
            ax.set_title(title)

        # Show values on heatmap
        if show_values:
            for i in range(len(y_labels)):
                for j in range(len(x_labels)):
                    val = attr_matrix[i, j]
                    color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                           color=color, fontsize=8)

        plt.tight_layout()
        return fig, ax

    def plot_feature_importance(
        self,
        attributions: Tensor,
        aggregation: str = 'l2_norm',
        temporal_aggregation: str = 'sum',
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        ax=None
    ):
        """
        Plot bar chart of feature importance.

        Args:
            attributions: Attribution tensor
            aggregation: How to aggregate across embedding dims
            temporal_aggregation: How to aggregate across time ('sum', 'mean', 'max')
            figsize: Figure size
            title: Plot title
            ax: Matplotlib axes

        Returns:
            Matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        # Aggregate by feature
        feature_attrs = self.aggregate_by_feature(attributions, aggregation)

        # Aggregate across time
        importance = {}
        for name, attr in feature_attrs.items():
            attr_np = attr.detach().cpu().numpy()
            if temporal_aggregation == 'sum':
                importance[name] = np.abs(attr_np).sum()
            elif temporal_aggregation == 'mean':
                importance[name] = np.abs(attr_np).mean()
            elif temporal_aggregation == 'max':
                importance[name] = np.abs(attr_np).max()
            else:
                raise ValueError(f"Unknown temporal aggregation: {temporal_aggregation}")

        # Sort by importance
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        # Create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Plot bars
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(names)))
        bars = ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Attribution Importance')
        ax.invert_yaxis()

        if title:
            ax.set_title(title)

        plt.tight_layout()
        return fig, ax

    def plot_multi_feature_heatmap(
        self,
        attributions: Tensor,
        prefix: Tuple[List[Tensor], List[Tensor]],
        features: Optional[List[str]] = None,
        aggregation: str = 'l2_norm',
        figsize: Tuple[int, int] = (14, 8),
        cmap: str = 'RdBu_r',
        title: Optional[str] = None
    ):
        """
        Plot heatmap with multiple features as rows.

        Args:
            attributions: Attribution tensor
            prefix: Original prefix data
            features: List of feature names to include (None for all)
            aggregation: Aggregation method
            figsize: Figure size
            cmap: Colormap
            title: Plot title

        Returns:
            Matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        # Aggregate by feature
        feature_attrs = self.aggregate_by_feature(attributions, aggregation)

        # Filter features if specified
        if features:
            feature_attrs = {k: v for k, v in feature_attrs.items() if k in features}

        if not feature_attrs:
            raise ValueError("No features to plot")

        # Build matrix
        feature_names = list(feature_attrs.keys())
        seq_len = next(iter(feature_attrs.values())).shape[0]
        attr_matrix = np.zeros((len(feature_names), seq_len))

        for i, name in enumerate(feature_names):
            attr_matrix[i] = feature_attrs[name].detach().cpu().numpy()

        # Get sequence labels
        x_labels = self._get_sequence_labels(prefix)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Normalize
        vmax = max(abs(attr_matrix.min()), abs(attr_matrix.max()))
        vmin = -vmax if vmax > 0 else -1

        # Plot
        im = ax.imshow(attr_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attribution')

        # Labels
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)

        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Feature')

        if title:
            ax.set_title(title)

        plt.tight_layout()
        return fig, ax

    def _get_sequence_labels(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        feature_idx: int = 0
    ) -> List[str]:
        """
        Get human-readable labels for sequence positions.

        Uses the first categorical feature (usually activity) for labeling.
        """
        cat_tensors, _ = prefix
        if not cat_tensors:
            return [f"t{i}" for i in range(cat_tensors[0].shape[1] if cat_tensors else 10)]

        # Use first categorical feature for labels
        indices = cat_tensors[feature_idx][0].detach().cpu().numpy()  # First batch item

        # Try to map to names
        if self.data_set_categories:
            cat_categories, _ = self.data_set_categories
            if feature_idx < len(cat_categories):
                feat_name = cat_categories[feature_idx][0]
                label_map = self._label_maps.get(feat_name, {})

                labels = []
                for idx in indices:
                    label = label_map.get(int(idx), str(idx))
                    # Truncate long labels
                    if len(label) > 15:
                        label = label[:12] + "..."
                    labels.append(label)
                return labels

        return [str(idx) for idx in indices]

    def to_text(
        self,
        attributions: Tensor,
        prefix: Tuple[List[Tensor], List[Tensor]],
        aggregation: str = 'l2_norm',
        width: int = 80
    ) -> str:
        """
        Create text-based visualization for terminal output.

        Args:
            attributions: Attribution tensor
            prefix: Original prefix data
            aggregation: Aggregation method
            width: Maximum width for bars

        Returns:
            String visualization
        """
        # Aggregate
        attr_values = self.aggregate_attributions(attributions, aggregation)
        attr_values = attr_values.detach().cpu().numpy()

        # Get labels
        labels = self._get_sequence_labels(prefix)

        # Normalize for bar width
        max_val = np.abs(attr_values).max()
        if max_val == 0:
            max_val = 1

        # Build text output
        lines = []
        lines.append("=" * width)
        lines.append("ATTRIBUTION SCORES")
        lines.append("=" * width)

        max_label_len = max(len(l) for l in labels)

        for i, (label, value) in enumerate(zip(labels, attr_values)):
            # Normalize to bar width
            bar_len = int(abs(value) / max_val * (width - max_label_len - 15))

            # Choose bar character based on sign
            if value >= 0:
                bar = "+" * bar_len
            else:
                bar = "-" * bar_len

            # Format line
            line = f"{label:>{max_label_len}} | {value:>8.4f} | {bar}"
            lines.append(line)

        lines.append("=" * width)

        return "\n".join(lines)

    def to_html(
        self,
        attributions: Tensor,
        prefix: Tuple[List[Tensor], List[Tensor]],
        aggregation: str = 'l2_norm',
        title: str = "Attribution Visualization"
    ) -> str:
        """
        Create interactive HTML visualization.

        Args:
            attributions: Attribution tensor
            prefix: Original prefix data
            aggregation: Aggregation method
            title: Page title

        Returns:
            HTML string
        """
        # Aggregate
        attr_values = self.aggregate_attributions(attributions, aggregation)
        attr_values = attr_values.detach().cpu().numpy()

        # Get labels
        labels = self._get_sequence_labels(prefix)

        # Normalize for color intensity
        max_val = np.abs(attr_values).max()
        if max_val == 0:
            max_val = 1

        # Build HTML
        html_parts = [f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .token {{
            padding: 10px 15px;
            border-radius: 5px;
            display: inline-block;
            margin: 5px;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .token:hover {{ transform: scale(1.1); }}
        .tooltip {{
            position: absolute;
            background: #333;
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 12px;
        }}
        h1 {{ color: #333; }}
        .legend {{
            margin-top: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .legend-bar {{
            width: 200px;
            height: 20px;
            background: linear-gradient(to right, #2166ac, white, #b2182b);
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="container">
"""]

        for label, value in zip(labels, attr_values):
            # Compute color based on attribution
            normalized = value / max_val  # -1 to 1

            if normalized >= 0:
                # Positive: white to red
                r = 255
                g = int(255 * (1 - normalized))
                b = int(255 * (1 - normalized))
            else:
                # Negative: white to blue
                r = int(255 * (1 + normalized))
                g = int(255 * (1 + normalized))
                b = 255

            color = f"rgb({r},{g},{b})"
            text_color = "white" if abs(normalized) > 0.5 else "black"

            html_parts.append(f'''
        <div class="token" style="background-color: {color}; color: {text_color};"
             title="Attribution: {value:.4f}">
            {label}
        </div>
''')

        html_parts.append("""
    </div>
    <div class="legend">
        <span>Negative</span>
        <div class="legend-bar"></div>
        <span>Positive</span>
    </div>
</body>
</html>
""")

        return "".join(html_parts)

    def save_html(
        self,
        attributions: Tensor,
        prefix: Tuple[List[Tensor], List[Tensor]],
        filepath: str,
        **kwargs
    ):
        """Save HTML visualization to file."""
        html = self.to_html(attributions, prefix, **kwargs)
        with open(filepath, 'w') as f:
            f.write(html)


class AttributionFlowDiagram:
    """
    Visualizes suffix prediction attributions as a flow diagram.

    This visualization separates case-level features (constant across events, e.g., Variant index)
    from event-level features (varying per event, e.g., Activity, Resource).

    Layout:
        - Top: Unified sequence (encoder prefix → SOS → suffix predictions)
        - Bottom: Case-level feature boxes

    Arrows:
        - Upward: From suffix predictions to earlier input positions (event-level attribution)
        - Downward: From suffix predictions to case-level features (case-level attribution)

    Arrow properties:
        - Thickness: Proportional to total attribution magnitude
        - Opacity: Proportional to attribution magnitude (0.15 to 1.0)
        - Color: Dominant feature at that position (argmax of event-level features)

    Example usage:
        >>> diagram = AttributionFlowDiagram(case_level_features=['Variant index', 'customer'])
        >>> fig = diagram.plot(results, feature_names, prefix_len=2, case_name="Case 123")
        >>> plt.show()
    """

    def __init__(self, case_level_features: Optional[List[str]] = None):
        """
        Initialize the AttributionFlowDiagram.

        Args:
            case_level_features: List of feature names that are constant across all events
                                in a case (e.g., ['Variant index', 'customer']).
                                If None, defaults to ['Variant index', 'customer'].
        """
        self.case_level_features = case_level_features or ['Variant index', 'customer']

    def plot(
        self,
        results: List[Tuple],
        feature_names: List[str],
        prefix_len: int,
        case_name: str,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        case_colormap: str = 'Purples',
        event_colormap: str = 'tab10'
    ):
        """
        Plot the attribution flow diagram.

        Args:
            results: List of tuples (suffix_step, attr_map, pred_desc, step_labels)
                    from InterpretabilityTool.compute_attribution_map()
            feature_names: List of all feature names
            prefix_len: Length of the encoder prefix
            case_name: Name of the case being visualized
            figsize: Figure size (width, height). If None, auto-calculated.
            title: Custom title. If None, auto-generated.
            case_colormap: Matplotlib colormap for case-level features
            event_colormap: Matplotlib colormap for event-level features

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.patches import FancyBboxPatch
        except ImportError:
            raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")

        event_level_features = [f for f in feature_names if f not in self.case_level_features]

        n_suffix = len(results)

        # Get labels from results
        _, _, _, first_step_labels = results[0]
        encoder_labels = first_step_labels[:prefix_len]

        # Suffix prediction labels
        suffix_labels = []
        for s_idx, (_, _, pred_desc, _) in enumerate(results):
            pred_label = pred_desc.split('=')[1].split('(')[0].strip() if '=' in pred_desc else f"pred_{s_idx}"
            suffix_labels.append(pred_label)

        # Total sequence length
        n_sequence = prefix_len + 1 + n_suffix  # encoder + SOS + suffix predictions

        # Color palettes
        case_colors = plt.cm.get_cmap(case_colormap)(np.linspace(0.4, 0.8, len(self.case_level_features)))
        case_feature_colors = {f: case_colors[i] for i, f in enumerate(self.case_level_features)}

        event_colors = plt.cm.get_cmap(event_colormap)(np.linspace(0, 1, len(event_level_features)))
        event_feature_colors = {f: event_colors[i] for i, f in enumerate(event_level_features)}

        # Figure setup
        if figsize is None:
            fig_width = max(14, n_sequence * 1.8 + 2)
            fig_height = 9
        else:
            fig_width, fig_height = figsize

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Layout coordinates
        seq_y = 6.5
        case_y = 2.0
        x_start = 1.0
        box_spacing = 1.5

        ax.set_xlim(-0.5, n_sequence * box_spacing + 2)
        ax.set_ylim(0.5, 9)
        ax.axis('off')

        # =====================================================================
        # 1. UNIFIED SEQUENCE (encoder + SOS + suffix predictions)
        # =====================================================================
        box_positions = []
        box_width = 1.1
        box_height = 0.7

        seq_idx = 0

        # Encoder positions
        for i, label in enumerate(encoder_labels):
            sx = x_start + seq_idx * box_spacing

            short_label = label.replace('enc(', '').replace(')', '')
            if len(short_label) > 10:
                short_label = short_label[:8] + '..'

            box = FancyBboxPatch((sx - box_width/2, seq_y - box_height/2),
                                 box_width, box_height,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor='#3498db', edgecolor='white', linewidth=2, alpha=0.9)
            ax.add_patch(box)

            ax.text(sx, seq_y, short_label, ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')
            ax.text(sx, seq_y + box_height/2 + 0.15, 'ENC', ha='center',
                   fontsize=7, color='#3498db', fontweight='bold')

            box_positions.append((sx, seq_y + box_height/2, seq_y - box_height/2, 'encoder'))
            seq_idx += 1

        # SOS token
        sx = x_start + seq_idx * box_spacing
        box = FancyBboxPatch((sx - box_width/2, seq_y - box_height/2),
                             box_width, box_height,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor='#9b59b6', edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(box)

        ax.text(sx, seq_y, "SOS", ha='center', va='center',
               fontsize=8, fontweight='bold', color='white')
        ax.text(sx, seq_y + box_height/2 + 0.15, 'DEC', ha='center',
               fontsize=7, color='#9b59b6', fontweight='bold')

        box_positions.append((sx, seq_y + box_height/2, seq_y - box_height/2, 'sos'))
        seq_idx += 1

        # Suffix predictions
        suffix_box_indices = []

        for s_idx, pred_label in enumerate(suffix_labels):
            sx = x_start + seq_idx * box_spacing

            short_label = pred_label if len(pred_label) <= 10 else pred_label[:8] + '..'

            box = FancyBboxPatch((sx - box_width/2, seq_y - box_height/2),
                                 box_width, box_height,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor='#2ecc71', edgecolor='white', linewidth=2, alpha=0.9)
            ax.add_patch(box)

            ax.text(sx, seq_y, short_label, ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')
            ax.text(sx, seq_y + box_height/2 + 0.15, f'PRED_{s_idx}', ha='center',
                   fontsize=7, color='#2ecc71', fontweight='bold')

            suffix_box_indices.append(len(box_positions))
            box_positions.append((sx, seq_y + box_height/2, seq_y - box_height/2, 'suffix'))
            seq_idx += 1

        # =====================================================================
        # 2. SEQUENCE FLOW ARROWS
        # =====================================================================
        for i in range(len(box_positions) - 1):
            x1, _, _, _ = box_positions[i]
            x2, _, _, _ = box_positions[i + 1]

            ax.annotate('',
                       xy=(x2 - box_width/2 - 0.05, seq_y),
                       xytext=(x1 + box_width/2 + 0.05, seq_y),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

        # Encoder/Decoder separator
        sep_x = x_start + (prefix_len - 0.5) * box_spacing
        ax.plot([sep_x, sep_x], [seq_y - box_height/2 - 0.3, seq_y + box_height/2 + 0.3],
               color='black', linewidth=2, linestyle='--')

        # =====================================================================
        # 3. CASE-LEVEL FEATURE BOXES
        # =====================================================================
        n_case = len(self.case_level_features)
        case_box_width = 1.8
        case_box_height = 0.7
        case_total_width = n_case * (case_box_width + 0.5)
        case_start_x = (n_sequence * box_spacing) / 2 - case_total_width / 2 + x_start

        case_box_positions = {}
        for i, feat in enumerate(self.case_level_features):
            cx = case_start_x + i * (case_box_width + 0.5) + case_box_width / 2

            color = case_feature_colors[feat]
            box = FancyBboxPatch((cx - case_box_width/2, case_y - case_box_height/2),
                                 case_box_width, case_box_height,
                                 boxstyle="round,pad=0.05,rounding_size=0.15",
                                 facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
            ax.add_patch(box)

            feat_label = feat if len(feat) <= 14 else feat[:12] + '..'
            ax.text(cx, case_y, feat_label, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')

            case_box_positions[feat] = (cx, case_y + case_box_height/2)

        ax.text((n_sequence * box_spacing) / 2 + x_start, case_y - case_box_height/2 - 0.3,
               'CASE-LEVEL FEATURES', ha='center', fontsize=11, fontweight='bold', color='#2c3e50')

        # =====================================================================
        # COMPUTE GLOBAL MAX FOR SCALING
        # =====================================================================
        global_max_event = 0.001
        global_max_case = 0.001

        for s_idx, (suffix_step, attr_map, pred_desc, step_labels) in enumerate(results):
            n_pos = len(step_labels)

            for p in range(n_pos):
                event_sum = sum(abs(attr_map.attributions[f][p].item()
                                   if hasattr(attr_map.attributions[f][p], 'item')
                                   else abs(attr_map.attributions[f][p]))
                              for f in event_level_features
                              if f in attr_map.attributions and p < len(attr_map.attributions[f]))
                global_max_event = max(global_max_event, event_sum)

            for feat in self.case_level_features:
                if feat in attr_map.attributions:
                    vals = attr_map.attributions[feat]
                    vals = vals.detach().cpu().numpy() if hasattr(vals, 'detach') else np.array(vals)
                    case_total = np.sum(np.abs(vals[:n_pos]))
                    global_max_case = max(global_max_case, case_total)

        # =====================================================================
        # DRAW ATTRIBUTION ARROWS: Event-level (upward)
        # =====================================================================
        for s_idx, (suffix_step, attr_map, pred_desc, step_labels) in enumerate(results):
            n_input_pos = len(step_labels)

            suffix_box_idx = suffix_box_indices[s_idx]
            sx, sy_top, sy_bottom, _ = box_positions[suffix_box_idx]

            for p in range(n_input_pos):
                event_sum = 0
                max_event_val = 0
                max_event_feat = None

                for feat in event_level_features:
                    if feat in attr_map.attributions and p < len(attr_map.attributions[feat]):
                        val = attr_map.attributions[feat][p]
                        val = val.item() if hasattr(val, 'item') else val
                        event_sum += abs(val)
                        if abs(val) > max_event_val:
                            max_event_val = abs(val)
                            max_event_feat = feat

                if event_sum < global_max_event * 0.02 or max_event_feat is None:
                    continue

                ratio = event_sum / global_max_event
                width = 0.5 + 4 * ratio
                alpha = 0.15 + 0.85 * ratio
                color = event_feature_colors.get(max_event_feat, 'gray')

                target_box_idx = p
                tx, ty_top, _, _ = box_positions[target_box_idx]

                ax.annotate('',
                           xy=(tx, ty_top + 0.05),
                           xytext=(sx, sy_top + 0.05),
                           arrowprops=dict(arrowstyle='->', color=color, lw=width,
                                          alpha=alpha, connectionstyle="arc3,rad=0.35",
                                          shrinkA=3, shrinkB=3))

        # =====================================================================
        # DRAW ATTRIBUTION ARROWS: Case-level (downward)
        # =====================================================================
        for s_idx, (suffix_step, attr_map, pred_desc, step_labels) in enumerate(results):
            n_pos = len(step_labels)

            suffix_box_idx = suffix_box_indices[s_idx]
            sx, sy_top, sy_bottom, _ = box_positions[suffix_box_idx]

            for feat in self.case_level_features:
                if feat not in attr_map.attributions:
                    continue

                vals = attr_map.attributions[feat]
                vals = vals.detach().cpu().numpy() if hasattr(vals, 'detach') else np.array(vals)
                case_total = np.sum(np.abs(vals[:n_pos]))

                if case_total < global_max_case * 0.02:
                    continue

                ratio = case_total / global_max_case
                width = 0.5 + 4 * ratio
                alpha = 0.15 + 0.85 * ratio
                color = case_feature_colors[feat]

                cx, cy_top = case_box_positions[feat]

                ax.annotate('',
                           xy=(cx, cy_top),
                           xytext=(sx, sy_bottom - 0.05),
                           arrowprops=dict(arrowstyle='->', color=color, lw=width,
                                          alpha=alpha, connectionstyle="arc3,rad=-0.15",
                                          shrinkA=3, shrinkB=3))

        # =====================================================================
        # LEGEND
        # =====================================================================
        case_patches = [mpatches.Patch(color=case_feature_colors[f], label=f)
                       for f in self.case_level_features]
        event_patches = [mpatches.Patch(color=event_feature_colors[f], label=f)
                        for f in event_level_features]

        legend1 = ax.legend(handles=case_patches, loc='lower left', title='Case-Level',
                           fontsize=7, title_fontsize=8, bbox_to_anchor=(0.01, 0.01))
        ax.add_artist(legend1)
        ax.legend(handles=event_patches, loc='lower right', title='Event-Level',
                 fontsize=7, title_fontsize=8, bbox_to_anchor=(0.99, 0.01))

        # Title
        if title is None:
            title = (f'Attribution Flow Diagram: {case_name}\n'
                    f'Arrows: suffix → input positions (up) & case-level features (down)\n'
                    f'Thickness & opacity = attribution magnitude | Color = dominant feature')

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        plt.tight_layout()
        return fig

    def save(
        self,
        results: List[Tuple],
        feature_names: List[str],
        prefix_len: int,
        case_name: str,
        filepath: str,
        dpi: int = 150,
        **kwargs
    ):
        """
        Save the attribution flow diagram to a file.

        Args:
            results: Attribution results from InterpretabilityTool
            feature_names: List of all feature names
            prefix_len: Length of the encoder prefix
            case_name: Name of the case
            filepath: Output file path (e.g., 'diagram.png', 'diagram.pdf')
            dpi: Resolution for raster formats
            **kwargs: Additional arguments passed to plot()
        """
        fig = self.plot(results, feature_names, prefix_len, case_name, **kwargs)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(fig)
