# Interpretability Module — Status Overview

## What's Implemented

### Core Attribution Methods (`attribution/`)
All four gradient/perturbation-based attribution methods are complete and functional:

| Method | File | Notes |
|--------|------|-------|
| **Integrated Gradients** | `integrated_gradients.py` | Multiple Riemann approximation methods, convergence delta validation |
| **SHAP** | `shap_explainer.py` | Sampling-based KernelSHAP for sequences, feature-level and (feature, timestep)-level aggregation |
| **ICE** | `ice_explainer.py` | Finite-differences sensitivity analysis, per-timestep or global |
| **LIME** | `lime_explainer.py` | Local linear surrogate with exponential kernel, Ridge regression |

### Model Wrapper (`model/`)
- **`IGModelWrapper`** (`model_wrapper.py`) — Enables separate encoder/decoder attribution. Supports three modes: encoder-only, SOS-only, combined. Uses soft-embedding trick for gradient flow through autoregressive decoder. Production-ready.
- **Target Selectors** (`target_selectors.py`) — Categorical (activity, resource) and numerical (time) target selection. Auto-resolution via argmax. Logit vs softmax.
- **Baseline Generators** (`baselines.py`) — Zero, padding-token, and mean-embedding baselines.

### Counterfactual / Perturbation Methods (`perturbation_methods/`)

| Method | Source | Status |
|--------|--------|--------|
| **REVISED+** | Stevens et al. (2024) | Complete. LSTM-VAE feasibility + MP-Declare plausibility. Latent elite-sampling search. Heavily adapted from the original — see `revised_plus/ADAPTATIONS.md` for details. |
| **LORELEY** | Huang et al. (2022) | Complete. Levenshtein neighborhood + genetic algorithm + decision tree surrogate. Rule-based explanations. |
| **GA Counterfactual** | Guidotti et al. (2024) adapted | Complete. Genetic algorithm with constraint-preserving crossover/mutation. Mutable vs immutable features. |
| **Declare Constraints** | `declare/` | Complete. Full checker + miner for 16 templates. Prefix-safety analysis. MP-Declare via RuM/jpype. |

### High-Level API
- **`InterpretabilityTool`** (`interpretability_tool.py`) — Unified entry point. Takes a process + prefix length + target, returns `AttributionMap` with tensor, string, and figure outputs.

### Visualization (`visualization/`)
- **`AttributionVisualizer`** — Heatmaps, bar charts, HTML. Multiple aggregation methods (L2 norm, sum, mean, abs_sum, abs_max).
- **`process_performance_map.py`** — Dataset-level attribution aggregation for process model discovery.

### Interactive Tool (`interactive_tool/`)
Streamlit web app with:
- Case selection, editing, and prediction with uncertainty
- Multi-method attribution comparison (real-time heatmaps)
- DFG (Directly-Follows Graph) analysis
- Pathway explorer
- REST API layer (`api.py`, `api_attribution.py`, `api_schemas.py`)
- React frontend components

Run: `streamlit run src/interpretability/interactive_tool/app.py`

### Utilities (`utils/`)
- **`TensorDecoder`** — Decodes integer-encoded tensors back to human-readable activity/resource names, inverse-transforms standardised numericals.
- **`SequenceFormatter`** / **`SequenceChange`** — Sequence formatting and diff-style comparison.

### Configuration (`config/`)
Dataset-specific configs for: Helpdesk, BPIC-17, Domestic Declarations.

### Experimental (`experimental/`)
- **`attribution_explorer.py`** — Gradio-based alternative UI for attributions.
- **`latent_autoencoder.py`** — Autoencoder for latent space exploration.

### Notebooks (`notebooks/`)
60+ notebooks across three datasets (helpdesk, bpic17, domestic_declarations):

| # | Topic | Description |
|---|-------|-------------|
| 00 | Dataset overview | Statistics, coverage |
| 01 | Single case attribution | Input -> attribution table for one case |
| 02 | Dataset attribution / process performance map | Feature importance across population |
| 03 | Encoder/decoder split & feature-value attribution | Separate enc/dec analysis; attributions keyed by feature values |
| 04 | Variant index explorer | Exploration by process variant |
| 05 | Attribution by correctness | Correct vs incorrect predictions |
| 06 | Suffix attribution heatmap | Visual attribution matrices for suffix steps |
| 09 | Best predictions heatmap | Top-confidence predictions |
| 10 | Prediction accuracy | Model performance analysis |
| 12 | Predicted vs actual attribution | Comparing attributions for predicted vs ground truth |
| 13 | Latent autoencoder | Latent space exploration |
| 14 | Creative suffix visualizations | Alternative visualization approaches |
| 15 | Variant frequency attribution | Attributions weighted by variant frequency |
| 16 | REVISED+ counterfactuals | Full pipeline: constraint mining (16a), VAE training (16b), search (16c) |
| 17 | GA counterfactual | Genetic algorithm counterfactuals |
| 18 | Aggregate counterfactuals | Multi-case counterfactual patterns |
| 19 | Feature sweep | Sensitivity to systematic feature changes |
| 20 | CART surrogate | Decision tree surrogate model |
| 21 | Pathway explorer | Process variant discovery |

---

## What's Missing / Open Items

### Known Architectural Issue
The CLAUDE.md notes an open conflict: **encoder and decoder embeddings are built separately**, which conflicts with IG logic when you want end-to-end gradient flow through shared embeddings. The `IGModelWrapper` works around this with its separate encoder/decoder attribution modes, but a unified embedding approach is not yet implemented. The legacy code in `src/aa_philipp/model_with_integrated_gradients/` was an earlier attempt at this.

### Not Implemented
- **Attention-based explanations** — No attention rollout, attention head analysis, or transformer-style interpretability (not directly applicable to LSTM but could be adapted for the hidden state).
- **Concept-based explanations** (e.g., TCAV / Testing with Concept Activation Vectors) — No higher-level concept attribution.
- **Global surrogate beyond CART** — Only decision tree surrogates exist (notebook 20). No rule list extraction, no global linear model, no symbolic regression.
- **Counterfactual method: DiCE** — The original REVISED+ paper uses DiCE's gradient-based approach; this repo replaced it with latent elite-sampling (documented in ADAPTATIONS.md). A direct DiCE implementation is not present.
- **Counterfactual method: NICE / Growing Spheres** — Other popular counterfactual frameworks are not implemented.
- **Formal evaluation of explanations** — No automated faithfulness metrics (e.g., sufficiency, comprehensiveness, deletion/insertion AUC), no user studies, no inter-method agreement scores. Notebooks do qualitative comparison but no systematic quantitative evaluation.
- **LORELEY for multi-feature sequences** — LORELEY currently operates primarily on activity sequences. Full multi-feature support (as in REVISED+ and GA) is not complete.
- **Temporal attribution aggregation** — `src/aa_philipp/notebooks/temporal_attribution_analysis.ipynb` exists as exploratory work but this hasn't been promoted to a proper module.

### Partial / Could Be Extended
- **SHAP** — Functional but less exercised in notebooks compared to IG. No dedicated notebook for SHAP-specific analysis (e.g., SHAP summary plots, dependency plots).
- **LIME** — Implemented but no dedicated notebooks. Less integration with the interactive tool compared to IG.
- **Interactive tool** — Works but the React frontend components are lightly documented. API layer is functional but has no OpenAPI spec or formal documentation.
- **Cross-dataset comparison** — Notebooks exist per dataset but there's no unified cross-dataset comparison notebook.
- **Hyperparameter sensitivity for counterfactuals** — No systematic ablation of counterfactual method parameters (population size, number of generations, kernel width, etc.).

### Legacy Code (`src/aa_philipp/`)
The `aa_philipp/` directory contains an older implementation of IG with Captum's `LayerIntegratedGradients`, plus saliency map notebooks, spread analysis, and suffix aggregation. This code is functional but has been superseded by the main `interpretability/` module. It remains useful as reference, especially `classes/spread_utils/scores.py` (weighted distance metrics) and the graph builder.
