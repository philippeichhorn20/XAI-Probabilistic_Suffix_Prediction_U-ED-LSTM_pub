"""
Attribution Explorer - Interactive Web Interface

A Gradio-based tool for exploring model attributions interactively.
Supports Integrated Gradients, SHAP, and ICE methods.

Usage:
    python attribution_explorer.py

    Or with custom paths:
    python attribution_explorer.py --model path/to/model.pkl --dataset path/to/dataset.pkl

Then open the URL shown in the terminal (usually http://localhost:7860)
"""

import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
from io import BytesIO
import base64
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path (need src/ on sys.path for `model.*` and `interpretability.*` imports below)
_HERE = os.path.dirname(os.path.abspath(__file__))                              # src/interpretability/experimental
_INTERP = os.path.dirname(_HERE)                                                # src/interpretability
_SRC = os.path.dirname(_INTERP)                                                 # src
_ROOT = os.path.dirname(_SRC)                                                   # project root
for p in (_HERE, _INTERP, _SRC, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Install with: pip install gradio")
    sys.exit(1)

from model.dropout_uncertainty_enc_dec_LSTM.dropout_uncertainty_model import DropoutUncertaintyEncoderDecoderLSTM
from interpretability import InterpretabilityTool


class AttributionExplorer:
    """Interactive attribution explorer with web interface."""

    def __init__(self):
        self.model = None
        self.dataset = None
        self.tool = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_maps = {}
        self.cat_features = []
        self.num_features = []
        self.n_samples = 0

    def load_model(self, model_path: str) -> str:
        """Load a model from file."""
        try:
            self.model = DropoutUncertaintyEncoderDecoderLSTM.load(model_path, dropout=0.0)
            self.model.eval()
            self.model = self.model.to(self.device)

            # Extract feature info
            cat_categories, num_categories = self.model.data_set_categories
            self.cat_features = [name for name, _, _ in cat_categories]
            self.num_features = [name for name, _, _ in num_categories]

            # Build label maps
            self.label_maps = {}
            for name, size, label_dict in cat_categories:
                self.label_maps[name] = {v: k for k, v in label_dict.items()}

            # Initialize interpretability tool
            self.tool = InterpretabilityTool(self.model, self.model.data_set_categories, device=self.device)

            return f"Model loaded successfully!\nCategorical features: {self.cat_features}\nNumerical features: {self.num_features}\nDevice: {self.device}"
        except Exception as e:
            return f"Error loading model: {str(e)}"

    def load_dataset(self, dataset_path: str) -> Tuple[str, gr.update]:
        """Load a dataset from file."""
        try:
            self.dataset = torch.load(dataset_path, weights_only=False)
            self.n_samples = len(self.dataset)

            return (
                f"Dataset loaded: {self.n_samples} samples",
                gr.update(maximum=self.n_samples - 1, value=0, interactive=True)
            )
        except Exception as e:
            self.n_samples = 0
            return f"Error loading dataset: {str(e)}", gr.update(interactive=False)

    def get_case_info(self, case_idx: int) -> str:
        """Get information about a selected case."""
        if self.dataset is None:
            return "Dataset not loaded"

        if case_idx is None or case_idx < 0 or case_idx >= self.n_samples:
            return "Invalid case index"

        try:
            sample = self.dataset[case_idx]
            cat_tensors, num_tensors = sample[0], sample[1]

            seq_len = cat_tensors[0].shape[0] if cat_tensors else num_tensors[0].shape[0]

            # Get activity sequence
            if 'Activity' in self.label_maps and cat_tensors:
                activity_idx = self.cat_features.index('Activity')
                activities = [self.label_maps['Activity'].get(int(v), f"Unk_{v}")
                             for v in cat_tensors[activity_idx].numpy()]
                # Find first non-padding
                start_idx = 0
                for i, a in enumerate(activities):
                    if not a.startswith('Unk_') and int(cat_tensors[activity_idx][i]) > 0:
                        start_idx = i
                        break
                activities = activities[start_idx:]
                activity_str = " → ".join(activities[:10])
                if len(activities) > 10:
                    activity_str += f" ... ({len(activities)} total)"
            else:
                activity_str = "N/A"

            return f"Case {case_idx}\nSequence length: {seq_len}\nActivities: {activity_str}"
        except Exception as e:
            return f"Error: {str(e)}"

    def compute_attributions(
        self,
        case_idx: int,
        prefix_length: int,
        target_feature: str,
        target_class: str,
        compute_ig: bool,
        compute_shap: bool,
        compute_ice: bool,
        compute_split: bool,
        n_steps: int,
        n_shap_samples: int,
        progress=gr.Progress()
    ) -> Tuple[plt.Figure, plt.Figure, plt.Figure, str, str]:
        """Compute selected attribution methods."""

        if self.model is None or self.dataset is None or self.tool is None:
            empty_fig = plt.figure()
            return empty_fig, empty_fig, empty_fig, "Error: Model or dataset not loaded", ""

        if case_idx is None or case_idx < 0 or case_idx >= self.n_samples:
            empty_fig = plt.figure()
            return empty_fig, empty_fig, empty_fig, "Error: Invalid case index", ""

        if not (compute_ig or compute_shap or compute_ice):
            empty_fig = plt.figure()
            return empty_fig, empty_fig, empty_fig, "Error: Select at least one method", ""

        try:
            sample = self.dataset[case_idx]
            process = (sample[0], sample[1])

            # Determine target class
            if target_class == "auto":
                target_class_val = "auto"
            else:
                try:
                    target_class_val = int(target_class)
                except:
                    target_class_val = "auto"

            results = {}
            split_info = "Requires IG to be selected" if not compute_ig else ("Not requested" if not compute_split else "Computing...")

            # Count methods to compute for progress
            n_methods = sum([compute_ig, compute_shap, compute_ice])
            progress_step = 0.9 / max(n_methods, 1)
            current_progress = 0.05

            # Compute Integrated Gradients
            if compute_ig:
                progress(current_progress, desc="Computing Integrated Gradients...")
                try:
                    attr_ig = self.tool.compute_attribution_map(
                        process=process,
                        prefix_length=prefix_length,
                        target=target_feature,
                        target_class=target_class_val,
                        suffix_scope="next",
                        method="integrated_gradients",
                        n_steps=n_steps
                    )
                    results['ig'] = attr_ig

                    # Compute encoder/decoder split (only if requested)
                    if compute_split:
                        actual_len, prefix = self.tool.extract_prefix_from_process(process, prefix_length)
                        split = self.tool.compute_encoder_decoder_attribution_split(
                            prefix=prefix,
                            target_output=target_feature,
                            target_value=target_class_val,
                            suffix_step=0,
                            n_steps=max(10, n_steps // 2)  # Use fewer steps for split
                        )
                        split_info = f"Encoder: {split['encoder_fraction']*100:.1f}% | Decoder SOS: {split['decoder_sos_fraction']*100:.1f}%"
                except Exception as e:
                    results['ig'] = None
                    split_info = f"IG Error: {str(e)}"
                current_progress += progress_step

            # Compute SHAP
            if compute_shap:
                progress(current_progress, desc="Computing SHAP...")
                try:
                    attr_shap = self.tool.compute_attribution_map(
                        process=process,
                        prefix_length=prefix_length,
                        target=target_feature,
                        target_class=target_class_val,
                        suffix_scope="next",
                        method="shap",
                        n_samples=n_shap_samples
                    )
                    results['shap'] = attr_shap
                except Exception as e:
                    results['shap'] = None
                current_progress += progress_step

            # Compute ICE
            if compute_ice:
                progress(current_progress, desc="Computing ICE...")
                try:
                    attr_ice = self.tool.compute_attribution_map(
                        process=process,
                        prefix_length=prefix_length,
                        target=target_feature,
                        target_class=target_class_val,
                        suffix_scope="next",
                        method="ice"
                    )
                    results['ice'] = attr_ice
                except Exception as e:
                    results['ice'] = None

            progress(0.95, desc="Generating plots...")

            # Create figures (pass not_computed flag for unselected methods)
            fig_ig = self._create_attribution_plot(results.get('ig'), "Integrated Gradients", not_computed=not compute_ig)
            fig_shap = self._create_attribution_plot(results.get('shap'), "SHAP", not_computed=not compute_shap)
            fig_ice = self._create_attribution_plot(results.get('ice'), "ICE Sensitivity", not_computed=not compute_ice)

            # Summary text
            summary = self._create_summary(results, case_idx, prefix_length, target_feature)

            progress(1.0, desc="Done!")

            return fig_ig, fig_shap, fig_ice, summary, split_info

        except Exception as e:
            empty_fig = plt.figure()
            return empty_fig, empty_fig, empty_fig, f"Error: {str(e)}", ""

    def _create_attribution_plot(self, attr_map, title: str, not_computed: bool = False) -> plt.Figure:
        """Create a heatmap plot for an attribution map."""
        fig, ax = plt.subplots(figsize=(12, 6))

        if attr_map is None:
            msg = f"{title}\nNot selected" if not_computed else f"{title}\nComputation failed"
            ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=14, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig

        matrix = attr_map.to_numpy()

        # Normalize for symmetric colormap
        vmax = np.abs(matrix).max()
        vmin = -vmax if vmax > 0 else -1

        im = ax.imshow(matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')

        # Labels
        step_labels = [str(l)[:12] for l in attr_map.step_labels]
        feature_labels = [str(f)[:18] for f in attr_map.feature_names]

        ax.set_xticks(range(len(step_labels)))
        ax.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(feature_labels)))
        ax.set_yticklabels(feature_labels, fontsize=9)

        ax.set_xlabel('Prefix Step')
        ax.set_ylabel('Feature')
        ax.set_title(f"{title}\n{attr_map.target_description}")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attribution')

        # Add values
        for i in range(len(feature_labels)):
            for j in range(len(step_labels)):
                val = matrix[i, j]
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       color=color, fontsize=6)

        plt.tight_layout()
        return fig

    def _create_summary(self, results: Dict, case_idx: int, prefix_len: int, target: str) -> str:
        """Create a text summary of the attribution results."""
        lines = []
        lines.append(f"Case: {case_idx} | Prefix Length: {prefix_len} | Target: {target}")
        lines.append("=" * 60)

        for method, attr_map in results.items():
            if attr_map is None:
                lines.append(f"\n{method.upper()}: Failed to compute")
                continue

            matrix = attr_map.to_numpy()
            total_attr = np.abs(matrix).sum()

            lines.append(f"\n{method.upper()}:")
            lines.append(f"  Total attribution: {total_attr:.4f}")

            if attr_map.convergence_delta is not None:
                lines.append(f"  Convergence delta: {attr_map.convergence_delta:.6f}")

            # Top features
            feature_sums = np.abs(matrix).sum(axis=1)
            top_idx = np.argsort(feature_sums)[::-1][:3]
            lines.append("  Top features:")
            for idx in top_idx:
                lines.append(f"    - {attr_map.feature_names[idx]}: {feature_sums[idx]:.4f}")

        return "\n".join(lines)

    def get_target_features(self) -> List[str]:
        """Get list of available target features."""
        return self.cat_features + self.num_features

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""

        with gr.Blocks(title="Attribution Explorer", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # Attribution Explorer

            Interactive tool for exploring model attributions using Integrated Gradients, SHAP, and ICE.

            **Instructions:**
            1. Load a model and dataset (.pkl files)
            2. Select a case and configure parameters
            3. Select which methods to compute (IG is fastest, SHAP/ICE are slower)
            4. Click "Compute Attributions"

            **Speed tips:** Start with just IG (fastest). Lower steps/samples = faster but less accurate.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Load Model & Data")

                    model_path = gr.Textbox(
                        label="Model Path",
                        value="../notebooks/training_variational_dropout/Helpdesk/Helpdesk_full_grad_norm_philipp_4layer.pkl",
                        placeholder="Path to model .pkl file"
                    )
                    load_model_btn = gr.Button("Load Model", variant="primary")
                    model_status = gr.Textbox(label="Model Status", interactive=False, lines=4)

                    dataset_path = gr.Textbox(
                        label="Dataset Path",
                        value="../../encoded_data/test_philipp/helpdesk_all_5_test.pkl",
                        placeholder="Path to dataset .pkl file"
                    )
                    load_dataset_btn = gr.Button("Load Dataset", variant="primary")
                    dataset_status = gr.Textbox(label="Dataset Status", interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### 2. Select Case & Configure")

                    case_selector = gr.Slider(
                        minimum=0, maximum=100, value=0, step=1,
                        label="Case Index",
                        interactive=False  # Disabled until dataset loaded
                    )
                    case_info = gr.Textbox(label="Case Info", interactive=False, lines=3)
                    load_case_btn = gr.Button("Load Case Info", size="sm")

                    prefix_length = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="Prefix Length"
                    )

                    target_feature = gr.Dropdown(
                        label="Target Feature",
                        choices=["Activity", "Resource"],
                        value="Activity",
                        interactive=True
                    )

                    target_class = gr.Textbox(
                        label="Target Class (integer or 'auto')",
                        value="auto",
                        placeholder="auto or class index (e.g., 3)"
                    )

                    gr.Markdown("**Methods to Compute:**")
                    with gr.Row():
                        compute_ig_cb = gr.Checkbox(label="Integrated Gradients", value=True)
                        compute_shap_cb = gr.Checkbox(label="SHAP", value=False)
                        compute_ice_cb = gr.Checkbox(label="ICE", value=False)
                        compute_split_cb = gr.Checkbox(label="Enc/Dec Split", value=True)

                    with gr.Row():
                        n_steps = gr.Slider(
                            minimum=10, maximum=100, value=20, step=10,
                            label="IG Steps (more = slower but more accurate)"
                        )
                        n_shap_samples = gr.Slider(
                            minimum=20, maximum=200, value=30, step=10,
                            label="SHAP Samples (more = slower but more accurate)"
                        )

            compute_btn = gr.Button("Compute Attributions", variant="primary", size="lg")

            with gr.Row():
                split_info = gr.Textbox(label="Encoder/Decoder Split (IG)", interactive=False)

            gr.Markdown("### 3. Attribution Results")

            with gr.Tabs():
                with gr.TabItem("Integrated Gradients"):
                    ig_plot = gr.Plot(label="Integrated Gradients")

                with gr.TabItem("SHAP"):
                    shap_plot = gr.Plot(label="SHAP")

                with gr.TabItem("ICE Sensitivity"):
                    ice_plot = gr.Plot(label="ICE")

                with gr.TabItem("Summary"):
                    summary_text = gr.Textbox(label="Summary", lines=20, interactive=False)

            with gr.Accordion("Comparison View", open=False):
                with gr.Row():
                    ig_plot_small = gr.Plot(label="IG")
                    shap_plot_small = gr.Plot(label="SHAP")
                    ice_plot_small = gr.Plot(label="ICE")

            # Event handlers
            load_model_btn.click(
                fn=self.load_model,
                inputs=[model_path],
                outputs=[model_status]
            ).then(
                fn=lambda: gr.update(choices=self.get_target_features(), value=self.cat_features[0] if self.cat_features else None),
                outputs=[target_feature]
            )

            load_dataset_btn.click(
                fn=self.load_dataset,
                inputs=[dataset_path],
                outputs=[dataset_status, case_selector]
            )

            load_case_btn.click(
                fn=self.get_case_info,
                inputs=[case_selector],
                outputs=[case_info]
            )

            compute_btn.click(
                fn=self.compute_attributions,
                inputs=[case_selector, prefix_length, target_feature, target_class,
                       compute_ig_cb, compute_shap_cb, compute_ice_cb, compute_split_cb,
                       n_steps, n_shap_samples],
                outputs=[ig_plot, shap_plot, ice_plot, summary_text, split_info]
            ).then(
                fn=lambda ig, shap, ice: (ig, shap, ice),
                inputs=[ig_plot, shap_plot, ice_plot],
                outputs=[ig_plot_small, shap_plot_small, ice_plot_small]
            )

        return interface


def main():
    parser = argparse.ArgumentParser(description="Attribution Explorer - Interactive Web Interface")
    parser.add_argument("--model", type=str, default=None, help="Path to model file to pre-load")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset file to pre-load")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    args = parser.parse_args()

    print("Starting Attribution Explorer...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    explorer = AttributionExplorer()

    # Pre-load model and dataset if provided
    if args.model:
        print(f"Pre-loading model: {args.model}")
        result = explorer.load_model(args.model)
        print(result)

    if args.dataset:
        print(f"Pre-loading dataset: {args.dataset}")
        result, _ = explorer.load_dataset(args.dataset)
        print(result)

    # Create and launch interface
    interface = explorer.create_interface()

    print(f"\nLaunching web interface on port {args.port}...")
    print(f"Open http://localhost:{args.port} in your browser")

    interface.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
