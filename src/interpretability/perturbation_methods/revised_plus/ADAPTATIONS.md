# REVISED+ Adaptations

**Original paper:** Stevens et al. (2024) - "Generating Feasible and Plausible Counterfactual Explanations for Outcome Prediction of Business Processes" (arXiv:2403.09232)

## Changes from Original

## 1. Declare constraints:


## Removed declaure constraints from training as it makes it much more computational expensive whil providing little beneift since it is not differenciable and the constraaints are learned anyway

### Declare -> MP-Declare

As our model does not just focus on activities, the simple declare trace conditions do not suffice for our use case.
We need one that takes into account the other features such as resource and temporal features as well.

Therefore we use -> MP-Declare from *Schönig, Di Ciccio, Maggi, Mendling (ICSOC 2016) — "Discovery of Multi-perspective Declarative Process Models"*

### No Declare Label Constraints (DLC)

Since we predict suffix/next event (not a case-level outcome), there are no meaningful labels to mine DLC from. The original paper mines DLC per outcome class, but that's circular for next-activity prediction — using model predictions as pseudo-labels would create a self-referential loop. Trace-level constraints (TLC) already ensure realistic sequences.

### Two-Tier Plausibility for Prefixes

Declare constraints are defined over complete traces, but our counterfactuals are *prefixes*. Not all templates can be conclusively checked on prefixes:

- **Prefix-safe** (monotonic violations — once violated, no suffix fixes it): INIT, ABSENCE, PRECEDENCE, ALTERNATE_PRECEDENCE, CHAIN_PRECEDENCE, NOT_SUCCESSION
- **Not prefix-safe** (suffix could resolve): EXISTENCE, RESPONSE, SUCCESSION, CHAIN_RESPONSE, CO_EXISTENCE, LAST, EXACTLY, and their alternating/chaining variants

We use a two-tier scoring system:
1. **`plausibility_definite`** (prefix-safe constraints only): Used as the penalty term in the combined optimization score during search. Only penalizes violations we are 100% sure about.
2. **`plausibility_optimistic`** (all constraints, treating prefix as complete): Used as a validity gate — if a counterfactual fails even this lenient check (threshold: `min_plausibility`), it is rejected.

### VAE Trained on Prefixes, Not Full Traces

The original REVISED+ trains its VAE on complete traces. Since our counterfactuals are alternative prefixes (the input to the next-activity predictor), we train the VAE on **random-length prefixes**: for each trace in each batch, we sample a prefix length k ~ Uniform(1, trace_length) and take the first k events. This means:
- The VAE learns the prefix manifold, not the full-trace manifold
- Decoded outputs are naturally prefix-shaped (events + padding)
- Length changes are valid counterfactuals (e.g., "one extra approval step")
- Feasibility measures "does this look like a realistic prefix?" rather than "...full trace?"

**Left-padded data handling**: Our encoded datasets use left-padding (padding zeros at the start, events at the end of the tensor). The collate_fn accounts for this: it extracts the last `trace_len` positions (the actual events), takes the first `k` of those, and left-pads the result back to `seq_len`. This ensures the VAE trains on properly structured prefixes.

**VAE persistence**: Trained VAEs are saved via `torch.save()` (whole model) to avoid retraining. Pass `vae_path` to `create_revised_plus_for_model()` — the VAE loads if the file exists, trains and saves otherwise.

### Latent Space Search Instead of Gradient-Based DiCE

The original paper uses DiCE's gradient-based optimization. Our data is discrete sequences and the prediction model is not differentiable through the VAE, so we use iterative elite-sampling in the VAE latent space instead: encode to z, sample perturbations, decode, score, refine toward the best candidates.

### Max Length Delta Constraint

Without constraints, the VAE can "cheat" by producing much shorter or longer sequences that trivially change the model's prediction. We add a `max_length_delta` parameter (default=2) that filters candidates whose prefix length deviates more than ±`max_length_delta` events from the original. This forces the search to find *content-level* edits (changing which activities occur) rather than *structural* shortcuts (just truncating the prefix).

### Proximity and Sparsity with Variable-Length Counterfactuals

Counterfactuals can have different prefix lengths than the original (within the `max_length_delta` bound). The proximity and sparsity metrics use a **union mask** — they consider positions where *either* the original or the counterfactual has events. This correctly captures both content changes and length changes.

The combined score normalizes proximity and sparsity by the **effective prefix length** (number of event positions in the union mask), not the full tensor length, so scores are meaningful regardless of how much padding exists.

## 2. Implementation

### Constraint Mining via RuM

We use RuM (Rule Mining Made Simple) in headless mode via jpype:

1. **MINERful** mines base Declare constraints (activity-based)
2. **MpEnhancer** adds data conditions on event attributes (Resource, seriousness, etc.)

```python
from src.interpretability.perturbation_methods import discover_mpdeclare

# Mine MP-Declare constraints with data conditions
constraints = discover_mpdeclare(
    'data/helpdesk.xes',
    min_support=0.9,
    data_conditions='ACTIVATIONS'  # or 'CORRELATIONS' or 'NONE'
)

# Constraints now include data conditions like:
# Response[Assign seriousness, Closed] | (seriousness is Value 1) ∧ (Activity is Assign seriousness)
```

### Data Condition Modes

- `ACTIVATIONS`: Mines conditions on activation event attributes
- `CORRELATIONS`: Mines correlations between activation and target attributes (may fail with timestamp attributes)
- `NONE`: Activity-only constraints (no data conditions)

