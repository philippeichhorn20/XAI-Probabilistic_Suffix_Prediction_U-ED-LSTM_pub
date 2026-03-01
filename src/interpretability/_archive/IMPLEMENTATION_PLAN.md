# Interpretability Deep Dive: Implementation Plan

## Overview

Three notebooks for comprehensive interpretability analysis of the U-ED-LSTM suffix prediction model.

---

## Prerequisites & Existing Components

| Component | Location | Status |
|-----------|----------|--------|
| Integrated Gradients | `src/interpretability/integrated_gradients.py` + Captum | ✅ Ready |
| SHAP | `src/interpretability/shap_explainer.py` (SequenceSHAP) | ✅ Ready |
| LIME | `src/interpretability/lime_explainer.py` | ❌ **To Implement** |
| Brier Score | `src/aa_philipp/classes/suffix_samples.py` | ✅ Ready |
| Shannon Entropy | `src/aa_philipp/classes/suffix_samples.py` | ✅ Ready |
| Attribution Visualization | `src/aa_philipp/notebooks/saliency_maps.ipynb` | ✅ Ready |
| Pre-computed Results | `results_store_helpdesk_results.pkl` | ✅ Available |
| IG Attribution Results | `results_store_integrated_gradients_helpdesk_*.pkl` | ✅ Available |

---

## New File: LIME Explainer

**File:** `src/interpretability/lime_explainer.py`

LIME (Local Interpretable Model-agnostic Explanations) will be implemented following the same pattern as SequenceSHAP:

```python
class SequenceLIME:
    """
    LIME explainer adapted for sequence models.

    1. Generate perturbed samples by masking/replacing features at each timestep
    2. Get model predictions for all perturbations
    3. Weight samples by proximity to original input (kernel function)
    4. Fit weighted linear regression to learn local feature importances
    """

    def __init__(self, model, data_set_categories, device='cpu'):
        ...

    def compute(
        self,
        prefix: Tuple[List[Tensor], List[Tensor]],
        target_output: str,
        target_value: Union[int, str] = 'auto',
        suffix_step: int = 0,
        n_samples: int = 500,
        kernel_width: float = 0.25,
        feature_level: bool = True
    ) -> Dict[str, Tensor]:
        ...
```

**Key Implementation Details:**
- Perturbation: Binary masking of (feature, timestep) pairs
- Kernel: Exponential kernel based on cosine distance
- Model: Ridge regression with L2 regularization
- Output: Feature importance weights per (feature, timestep) or aggregated per feature

---

## Notebook 1: Single Attribution Analysis

**File:** `src/interpretability/notebooks/01_single_attribution_analysis.ipynb`

### Section 1a: Top/Bottom Prefixes by Brier Score

**Goal:** Show how attribution patterns differ between accurate and inaccurate predictions.

**Implementation Steps:**
1. Load pre-computed results from `results_store_helpdesk_results.pkl`
2. Calculate Brier scores using `SuffixSamples.calculate_brier_score_full_suffix()`
3. Select top-5 (best, Brier ≈ 0) and bottom-5 (worst, Brier ≈ 1) prefixes
4. For each selected prefix:
   - Load or compute IG attributions for next-event activity prediction
   - Generate heatmap visualization (Feature × Timestep)
   - Show prefix events, true suffix, and mean prediction for context

**Visualization:**
```
┌─────────────────────────────────────────┐
│  Attribution Heatmap (Best Performer)   │
│  ┌─────────────────────────────────┐    │
│  │ Activity  █████ ███ ██████████  │    │
│  │ Resource  ██ ████ █████ ██████  │    │
│  │ Time      ███ ██ ████████ ████  │    │
│  └─────────────────────────────────┘    │
│  Prefix: [Event1] → [Event2] → [Event3] │
│  True Next: "Resolve ticket"            │
│  Predicted: "Resolve ticket" ✓          │
│  Brier Score: 0.0                       │
└─────────────────────────────────────────┘
```

### Section 1b: Feature Importance by Prediction Quality (Normalized)

**Goal:** Aggregate view showing if certain features are more important for accurate predictions.

**Implementation Steps:**
1. Bin all prefixes by Brier score quartiles (Q1: best, Q4: worst)
2. For each bin, compute mean absolute attribution per feature type
3. Normalize attributions within each bin (sum to 1)
4. Create grouped bar chart: Feature importance by quality quartile

**Visualization:**
```
Feature Importance by Prediction Quality
┌────────────────────────────────────────┐
│  █ Activity  █ Resource  █ Time       │
│                                        │
│  ███                                   │
│  ███ ██                                │
│  ███ ██ █    ███                       │
│  ███ ██ █    ███ ██                    │
│  ███ ██ █    ███ ██ █    ███ ██ █     │
│  ─────────   ─────────   ─────────    │
│  Q1 (Best)   Q2-Q3       Q4 (Worst)   │
└────────────────────────────────────────┘
```

### Section 1c: Top/Bottom Prefixes by Certainty (Entropy)

**Goal:** Show attribution patterns for certain vs uncertain predictions.

**Implementation Steps:**
1. Calculate Shannon Entropy using `calculate_shannon_entropy_on_activity_suffix()`
2. Select top-5 (lowest entropy = most certain) and bottom-5 (highest entropy = least certain)
3. Generate attribution heatmaps (same as 1a)
4. Compare patterns between certain/uncertain predictions

**Key Metric from spread_analysis:**
- Shannon Entropy strongly correlates with suffix length (-0.59 with prefix length)
- Control for suffix length when selecting samples to avoid confounding

---

## Notebook 2: Aggregated Attribution Analysis

**File:** `src/interpretability/notebooks/02_aggregated_attribution_analysis.ipynb`

### Section 2a: Attribution Impact on Performance

**Goal:** Determine if attribution on certain input factors correlates with prediction performance.

**Implementation Steps:**
1. Compute/load IG attributions for representative sample (use existing results or sample ~500 prefixes)
2. Aggregate attributions by:
   - **Feature type:** Activity, Resource, Time
   - **Temporal position:** First event, last event, relative position (%)
   - **Feature value:** Specific activity types receiving highest attribution
3. Build regression model:
   ```
   Brier_Score ~ Attribution_Activity + Attribution_Resource + Attribution_Time
                 + prefix_length + suffix_length
   ```
4. Report coefficients, p-values, R²
5. Visualizations:
   - Scatter plots: Attribution magnitude vs Brier score
   - Heatmap: Correlation matrix of attribution features and performance

**Analysis Questions:**
- Does high attribution on recent events correlate with better predictions?
- Do predictions fail when attribution is spread evenly vs concentrated?
- Is resource attribution more predictive of accuracy than activity attribution?

### Section 2b: Attribution Impact on Certainty

**Goal:** Can attribution patterns predict model uncertainty?

**Implementation Steps:**
1. Use same attribution data as 2a
2. Build regression model:
   ```
   Shannon_Entropy ~ Attribution_Activity + Attribution_Resource + Attribution_Time
                     + prefix_length + (prefix_length controls for confounding)
   ```
3. Additional analysis:
   - Attribution concentration (Gini coefficient of attribution distribution)
   - Attribution temporal skew (are recent events more attributed for certain predictions?)
4. Visualizations:
   - Scatter: Attribution concentration vs Entropy
   - Box plots: Attribution distribution for high/low certainty groups

**Hypothesis to Test:**
- Concentrated attribution (model "knows what to look at") → lower uncertainty
- Diffuse attribution (model "unsure where to look") → higher uncertainty

---

## Notebook 3: Method Comparison (IG vs SHAP vs LIME)

**File:** `src/interpretability/notebooks/03_method_comparison.ipynb`

### Setup

**Sample Selection:**
- Use 100-200 prefixes for computational feasibility
- Stratified by prefix length and Brier score to ensure diversity

### Section 3a: Compute Attributions with All Methods

```python
# For each prefix:
ig_attributions = compute_integrated_gradients(model, prefix, target_fn, n_steps=50)
shap_attributions = shap_explainer.compute(prefix, target_output='Activity', n_samples=100)
lime_attributions = lime_explainer.compute(prefix, target_output='Activity', n_samples=500)
```

### Section 3b: Rank Correlation Analysis

**Goal:** Do methods agree on feature importance rankings?

**Implementation:**
1. For each prefix, rank features by attribution magnitude
2. Compute pairwise Spearman rank correlation:
   - IG vs SHAP
   - IG vs LIME
   - SHAP vs LIME
3. Aggregate correlations across all samples

**Expected Output:**
```
Pairwise Rank Correlation (Spearman ρ)
┌──────────┬───────┬───────┬───────┐
│          │  IG   │ SHAP  │ LIME  │
├──────────┼───────┼───────┼───────┤
│ IG       │ 1.000 │ 0.XXX │ 0.XXX │
│ SHAP     │ 0.XXX │ 1.000 │ 0.XXX │
│ LIME     │ 0.XXX │ 0.XXX │ 1.000 │
└──────────┴───────┴───────┴───────┘
```

### Section 3c: Feature-Level Agreement

**Goal:** Do methods agree on which feature TYPE matters most?

**Implementation:**
1. For each prefix, identify the most important feature type (Activity/Resource/Time)
2. Compute agreement rate between method pairs
3. Identify cases of disagreement for qualitative analysis

### Section 3d: Qualitative Case Studies

**Goal:** Deep dive into cases where methods agree/disagree

**Implementation:**
1. Select 3-5 cases where all methods agree (high consensus)
2. Select 3-5 cases where methods disagree (low consensus)
3. For each case, show:
   - Side-by-side attribution heatmaps
   - Prefix context and prediction outcome
   - Discussion of why methods might differ

**Visualization:**
```
Case Study: Disagreement Example
┌───────────────────────────────────────────────────┐
│ Prefix: [Assign seriousness] → [Take in charge]   │
│ True Next: "Resolve ticket"                       │
├─────────────┬─────────────┬─────────────┬─────────┤
│             │     IG      │    SHAP     │   LIME  │
├─────────────┼─────────────┼─────────────┼─────────┤
│ Activity    │    0.65 ★   │    0.42     │   0.38  │
│ Resource    │    0.20     │    0.45 ★   │   0.28  │
│ Time        │    0.15     │    0.13     │   0.34 ★│
└─────────────┴─────────────┴─────────────┴─────────┘
★ = Most important feature according to method
```

### Section 3e: Computational Cost Comparison

| Method | Samples/Steps | Avg Runtime (per prefix) | Memory |
|--------|---------------|--------------------------|--------|
| IG     | 50 steps      | ~X.X sec                 | Low    |
| SHAP   | 100 samples   | ~X.X sec                 | Medium |
| LIME   | 500 samples   | ~X.X sec                 | Medium |

---

## File Structure

```
src/interpretability/
├── integrated_gradients.py      # Existing
├── shap_explainer.py            # Existing
├── lime_explainer.py            # NEW - To implement
├── visualization.py             # Existing
├── model_wrapper.py             # Existing
└── notebooks/
    ├── 01_single_attribution_analysis.ipynb      # NEW
    ├── 02_aggregated_attribution_analysis.ipynb  # NEW
    └── 03_method_comparison.ipynb                # NEW
```

---

## Implementation Order

1. **LIME Explainer** (`lime_explainer.py`) - Required for Notebook 3
2. **Notebook 1** - Can start immediately with existing IG implementation
3. **Notebook 2** - Builds on Notebook 1 analysis
4. **Notebook 3** - Requires LIME implementation

---

## Data Dependencies

| Notebook | Required Data Files |
|----------|---------------------|
| 1, 2, 3  | `results_store_helpdesk_results.pkl` (prefix, suffix, predictions) |
| 1, 2     | `results_store_integrated_gradients_helpdesk_*.pkl` (pre-computed IG) |
| 1, 2, 3  | Model checkpoint: `Helpdesk_full_grad_norm_philipp_4layer.pkl` |
| 1, 2, 3  | Test dataset: `helpdesk_all_5_test.pkl` |

---

## Notes

- **Embedding conflict:** Current IG implementation has a workaround in `saliency_maps.ipynb` line: `attributions[0][17, :, :19] += attributions[1][0]`. This merges encoder and decoder attributions. Need to document this clearly in notebooks.
- **Brier Score interpretation:** 0.0 = perfect prediction (true suffix always predicted), 1.0 = never predicted true suffix
- **Shannon Entropy:** Higher = more uncertain/spread predictions
- **Performance baseline from spread_analysis:** Correlation between Brier and Entropy is ~0.8+, both strongly negatively correlated with prefix length (~-0.59)
