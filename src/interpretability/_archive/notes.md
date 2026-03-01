# Notes

## Perturbation-based methods:
### LORELEY:
**Overview:** The LORELEY approach generates a neighborhood of the initial input sequence by applying a genetic algorithm that takes items from other samples from the dataset based on a fitness function and replaces parts of the original sequence with that of a different one, repeating this a number of times with given probabilities. To ensure adherence to control flow constraints, it treats the activity flow as a single indivisible property of each sequence.

**My criticism:** LORELEY does allow control flow to vary, but only by swapping entire control flow vectors atomically (treating them as one unit) rather than editing individual activities. This means it doesn't model cross-attribute constraints (e.g., which feature values are only valid given a particular activity sequence), which disregards a lot of possibilities to adapt the sequence that may be allowed, and includes a lot of ones that are not allowed through impossible activity ↔ feature combinations.

**Conclusion:** Looking into approaches that use more nuanced constraints. (e.g. revised)

### REVISED+:
**Overview:** Stevens et al. (2024) — VAE for feasibility + Declare constraints for plausibility. Generates counterfactuals that change the model's prediction while remaining realistic.

**Our adaptations:**
- No DLC (label constraints) — next-activity prediction has no case-level outcome labels, so DLC mining would be circular.
- Two-tier plausibility: prefix-safe constraints (INIT, ABSENCE, PRECEDENCE, etc.) for the search penalty; all constraints optimistically for the validity gate. This avoids false penalties from forward-looking constraints (RESPONSE, SUCCESSION) that can't be conclusively checked on prefixes.
- VAE trained on random-length prefixes (not full traces) with left-padding-aware truncation. Decoded outputs are naturally prefix-shaped and length changes are valid counterfactuals.
- Proximity/sparsity use union mask (positions where either original or CF has events) and normalize by effective prefix length, not tensor length.
- Max length delta constraint (default ±2): prevents the VAE from "cheating" by producing drastically shorter/longer sequences that trivially change predictions. Forces content-level edits.
- Latent space elite-sampling instead of gradient-based DiCE (discrete data, non-differentiable path).
- VAE persistence: trained models are saved as pkl files to avoid retraining.