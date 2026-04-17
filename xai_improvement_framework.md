# XAI-Driven Model Improvement Framework

A diagnostic framework connecting interpretability methods to actionable model improvements for sequence prediction models. Organized by XAI method: run the method, look for the listed signals, diagnose the issue, take action.

Each signal is tagged with its improvement category: **[Data]** = data augmentation, **[Arch]** = architectural improvement, **[Error]** = error detection.

---

## 1. Integrated Gradients — Dataset-Level Aggregate

Compute IG attributions across the test set and aggregate per feature.

### Signal 1.1 — Low-attribution features signal training gaps **[Data]**

- **Observation:** A subset of input features consistently receives near-zero attribution despite being semantically relevant to the prediction task.
- **Diagnosis:** The model has not learned to leverage these features — either because they lack variation in the training data, or because a few dominant features absorb all predictive signal.
- **Action:** Augment training data with cases where underrepresented features are predictive; increase feature diversity; consider feature-specific loss weighting.

> **Example — Helpdesk** (Notebook 02, 50 samples from 3,329 available, 16 features): 10 of 16 features have attribution < 0.04 (responsible_section, support_section, service_type, service_level, seriousness_2, seriousness, day_in_week, workgroup, seconds_in_day, product). Activity (0.422) and Variant Index (0.208) dominate.
>
> **Example — BPIC17** (Notebook 02, 200 test samples, 18 features): Financial features have near-zero attribution despite domain relevance — MonthlyCost (0.010), NumberOfTerms (0.006), FirstWithdrawalAmount (0.004). concept:name (1.985) dominates. Both datasets show the same pattern of few dominant features and many near-zero, but the specific ignored features differ.

### Signal 1.2 — Rare variants are memorized, not generalized **[Data]**

- **Observation:** Strong inverse correlation between variant frequency and attribution magnitude — rare variants receive disproportionately high attribution.
- **Diagnosis:** The model memorizes rare variants by relying heavily on their identity rather than learning generalizable structural patterns. Alternatively, if rare variants show near-zero attribution, the model ignores them entirely and defaults to majority-class behavior. Both patterns indicate data insufficiency.
- **Action:** Augment rare variants to move them above a minimum support threshold; use the frequency-attribution curve breakpoint to determine where memorization begins.

> **Example — Helpdesk** (Notebook 15, 57 unique variants, most common = 493 cases, many < 5): Strong inverse correlation (r = -0.78, p < 0.0001) between variant frequency and attribution magnitude.

---

## 2. Integrated Gradients — Encoder-Decoder Split

Apply IG separately to encoder hidden state and decoder SOS embedding to decompose their relative contributions.

### Signal 2.1 — Encoder-decoder balance varies across datasets and model configurations **[Arch]**

- **Observation:** The relative contribution of encoder vs. decoder varies across setups — driven by both the data and the decoder input dimensionality. Per-feature splits can differ even more than the aggregate.
- **Diagnosis:** Strongly encoder-dominant → decoder underutilized. Balanced → both paths contribute. Per-feature imbalances reveal which output heads could benefit from architectural changes.
- **Action:** For encoder-dominant: richer decoder input or attention. For per-feature imbalances: output-specific adaptations.

> **Example — Helpdesk** (Notebook 03, 100 samples; decoder input dim=19): 80.2% encoder / 19.8% decoder. Activity is balanced (54.6%/45.4%), time features slightly decoder-dominant (46.3%/53.7%).
>
> **Example — BPIC17** (Notebook 03, 100 samples; decoder input dim=44): 59.1% encoder / 40.9% decoder — more balanced, likely because the decoder receives richer input (44 vs. 19 dimensions). lifecycle:transition is decoder-dominant (33.8%/66.2%). Both models share the same core LSTM (4 layers, hidden=128).

### Signal 2.2 — Prefix length effect on encoder contribution **[Arch]**

- **Observation:** The encoder's relative contribution either stays constant, increases, or decreases as prefix length grows.
- **Diagnosis:** Constant → structural bottleneck. Increasing → the architecture benefits from longer context. Decreasing → longer prefixes introduce noise or the encoder loses information.
- **Action:** For structural bottleneck: attention over all encoder hidden states or Transformer-style cross-attention. For capacity bottleneck: wider or deeper encoder. For noise accumulation: gated input filtering or selective attention.

> **Example — Helpdesk** (Notebook 03): Encoder contribution decreases with prefix length (correlation −0.46), from ~90% at prefix lengths 1–2 down to ~50% at prefix length 6. The decoder SOS token becomes more important for longer sequences, suggesting the encoder loses information over longer prefixes.
>
> **Example — BPIC17** (Notebook 03): No strong relationship between prefix length and encoder contribution (correlation −0.26). Values range from 29.2% (prefix 49) to 79.5% (prefix 15) with high variance, likely due to small sample sizes at longer prefixes.

---

## 3. Integrated Gradients — Per-Value Analysis

Compute IG separately for each value a feature can take, and for value combinations.

### Signal 3.1 — High attribution variance within features **[Arch]**

- **Observation:** Attribution magnitude varies by an order of magnitude between different values of the same feature. Some feature-value pairs co-contribute strongly.
- **Diagnosis:** Uniform embedding dimensions waste capacity on low-impact values while constraining high-impact ones.
- **Action:** Non-uniform embedding allocation; hierarchical embeddings grouping semantically related values; joint embeddings for co-contributing feature-value pairs.

> **Example — Helpdesk** (Notebook 03b, 300 test samples): 10x attribution variance between values of the same feature. Top values: Variant index=5 (0.305), Variant index=52 (0.267). Top co-contributing pairs: Variant index + customer combinations (up to 0.826 combined).
>
> **Example — BPIC17** (Notebook 03b): With 28 activities, 151 resources, and 16 loan goals, the feature value space is much larger. Natural activity groupings (W_, A_, O_ prefixes) are strong candidates for hierarchical embeddings.

---

## 4. Integrated Gradients — Per-Case and Comparative

Apply IG to individual cases, compare attributions between correct/incorrect predictions, or between predicted and actual classes.

### Signal 4.1 — Classifiable error patterns **[Error]**

- **Observation:** Errors fall into identifiable categories based on their attribution profile: dispersed low-magnitude (no feature is informative), single-feature dominance (tunnel vision), or recency concentration (only the most recent events matter).
- **Diagnosis & Action:**
  - Dispersed → out-of-distribution input; add an abstain option or flag for human review
  - Tunnel vision → over-reliance on one feature; apply regularization or multi-task loss
  - Recency → positional bias; add positional encoding or attention over the full sequence
- **Cross-pattern insight:** The dominant error pattern may differ across datasets. Datasets with fewer dominant features tend toward tunnel vision; datasets with many informative features tend toward dispersed errors.

> **Example — Helpdesk** (Notebook 01, Case 1016, prefix=3): Activity attribution = 0.559, Resource = 0.183, convergence delta = 0.0016. Strong single-feature dominance pattern — Activity alone accounts for a disproportionate share of the total attribution.
>
> **Example — BPIC17** (Notebook 01, Application_1000086665, prefix=3): Top features are more diverse — concept:name (1.204), org:resource (0.924), lifecycle:transition (0.464), case_elapsed_time (0.181), Action (0.177). Attribution is spread across multiple features rather than concentrated in one, suggesting BPIC17 cases lean more toward the dispersed pattern.

### Signal 4.2 — Single-feature cause in misclassifications **[Error]**

- **Observation:** When computing IG targeted at predicted class minus IG targeted at actual class, the attribution delta is concentrated in one or two features.
- **Diagnosis:** Clear single-feature cause — the model ignored or misweighted one critical input. If the delta is instead dispersed, the cause is failed signal integration (an architecture problem).
- **Action:** For concentrated delta: investigate whether the responsible feature is underrepresented, mislabeled, or needs higher loss weight. For dispersed delta: consider architectural changes to improve signal integration.

> **Example — Helpdesk** (Notebook 12, Case 1261, predicted Take in charge ticket, actual Resolve ticket): Delta concentrated in Activity (-0.373) and Variant index (-0.336). All top 8 features push away from the correct class.

---

## 5. GA Counterfactual

Generate counterfactuals using a genetic algorithm with constraint-preserving crossover; analyze both individual and aggregate results.

### Signal 5.1 — Counterfactual transitions expose class imbalance **[Data]**

- **Observation:** Prediction transitions between classes are asymmetric — some require minimal changes while others require many, and certain class pairs appear far more frequently.
- **Diagnosis:** Class-specific data imbalance at the decision boundary; fragile separation between frequently confused pairs.
- **Action:** Targeted oversampling of the underrepresented side of asymmetric transitions; generate synthetic training data along the counterfactual directions.

> **Example — Helpdesk** (Notebook 18, 100 prefixes, 500 CFs): 8 unique prediction transitions observed; some need only 1 feature change.
>
> **Example — BPIC17** (Notebook 18, 10 prefixes, 50 CFs): 10/10 prefixes have at least one valid counterfactual. Significantly slower execution (1,020s) reflects the larger vocabulary and longer sequences.

### Signal 5.2 — Per-feature mutation scope reveals training support gaps **[Data]**

- **Observation:** Some features can flip predictions with single-feature mutations while others cannot, revealing which features the model has built fragile vs. robust representations for.
- **Diagnosis:** Features with easily flipped representations are fragile; features that cannot flip predictions alone may be underutilized.
- **Action:** Augment with cases that diversify the fragile features; consider feature-specific embeddings or loss weighting for underutilized features.

> **Example — Helpdesk** (Notebook 17): Activity-only mutations easily flip predictions; resource mutations alone do not — the model's activity representations are fragile while resource information is underutilized.
>
> **Example — BPIC17** (Notebook 17, 6,301 training sequences, 147 Declare constraints, 4 mutable features): The much larger resource vocabulary (151 values vs. Helpdesk's smaller set) means per-resource training support is thinner. Constraint-preserving crossover is essential — 147 mined constraints prevent generation of impossible sequences.

---

## 6. CART Surrogate

Train decision trees globally on the model's predictions and/or per decision point (gateway).

### Signal 6.1 — Surrogate disagreement reveals spurious patterns **[Error]**

- **Observation:** At some cases, the surrogate predicts correctly while the model does not, despite the surrogate being a simpler approximation.
- **Diagnosis:** The model has learned spurious patterns that a simpler, more constrained model avoids.
- **Action:** Regularize the model; use tree rules as inductive bias; apply knowledge distillation for the simple cases where the surrogate outperforms.

> **Example — Helpdesk** (Notebook 20, 4,245 prefixes, 916 cases): CART surrogate achieves 91.1% fidelity at depth=8 with 49 leaves. Top features: total_elapsed_time, last_activity, prefix_length, variant_index.
>
> **Example — BPIC17** (Notebook 20, 56,679 prefixes from 1,500 cases): Much larger surrogate training set. Prefix lengths range from 1 to 96, testing whether the model's advantage over trees grows with sequence length.

### Signal 6.2 — Three-way gateway triage classifies each decision point **[Arch]**

- **Observation:** Comparing LSTM accuracy, surrogate fidelity, and ground truth tree accuracy per gateway creates three distinct categories:
  - **Replaceable:** High fidelity + high LSTM accuracy + high GT accuracy → the gateway is simple, the LSTM gets it right, and a tree could replace it.
  - **Wrong pattern:** High fidelity + low LSTM accuracy → the surrogate perfectly captures what the LSTM does, but what the LSTM does is wrong. The model has learned incorrect decision logic at this gateway.
  - **LSTM advantage:** LSTM accuracy substantially exceeds GT accuracy → the LSTM has learned something a tree cannot express. Its complexity pays off here.
- **Diagnosis:** Not all gateways need the same treatment. Replaceable gateways waste model capacity. Wrong-pattern gateways are the highest-priority targets for retraining. LSTM-advantage gateways validate the architecture.
- **Action:** Route replaceable gateways to lightweight classifiers. For wrong-pattern gateways: extract the disagreement prefixes (where surrogate and GT tree disagree) as a targeted retraining set. For LSTM-advantage gateways: keep the full model and investigate what nonlinear patterns it captures.

> **Example — Helpdesk** (Notebook 22, 4 qualifying gateways): "Resolve ticket" is replaceable (93% LSTM acc, 92% fidelity, 82% GT acc). "Assign seriousness" is wrong-pattern (51% LSTM acc, 83% fidelity, 79% GT acc, only 22% surrogate-GT agreement — the LSTM has learned fundamentally different decision logic than the actual process).
>
> **Example — BPIC17** (Notebook 22, 21 qualifying gateways): "W_Handle leads" is replaceable (86% LSTM acc, 99% fidelity, 85% agreement). "A_Create Application" is wrong-pattern (24% LSTM acc, 100% fidelity, 100% GT acc, 24% agreement — the LSTM perfectly replicates wrong logic). "A_Complete" shows LSTM advantage (97% LSTM acc vs. 57% GT acc — the tree cannot capture the decision). "W_Assess potential fraud" is a complete failure (0% LSTM acc) caused by class imbalance rather than wrong reasoning.

### Signal 6.3 — Correct-vs-incorrect attribution isolates error-driving features per gateway **[Error]**

- **Observation:** Computing IG attributions separately for correct and incorrect predictions at the same gateway reveals which features the model systematically overweights or underweights when it makes mistakes.
- **Diagnosis:** Overweighted features in errors are candidates for spurious correlations at that gateway. Underweighted features are candidates for information the model should use but ignores.
- **Action:** For overweighted features: ablation study or regularization targeting that feature at that gateway. For underweighted features: augment with cases where that feature is predictive, or add feature-specific attention.

> **Example — Helpdesk** (Notebook 23, 4 gateways, 100 IG samples each): At "Take in charge ticket," the model underweights Activity by −0.33 when wrong — it fails to use the most informative feature. At "Resolve ticket," the model overweights Variant Index and Activity on its rare errors — it over-commits to identity features.
>
> **Example — BPIC17** (Notebook 23, 4 of 21 gateways completed): At "A_Create Application," wrong predictions overweight org:resource (+0.46) and underweight concept:name (−0.15) — the model focuses on who handled the case rather than what activity occurred. At "W_Handle leads," wrong predictions overweight concept:name (+0.20) and underweight org:resource (−0.21) — the opposite pattern, showing that error-driving features are gateway-specific.

---

## 7. Confusion Matrix

Standard prediction accuracy and confidence analysis over the test set.

### Signal 7.1 — Class imbalance directs analysis effort **[Error]**

- **Observation:** The model systematically fails on rare classes while achieving high accuracy on frequent ones. A single confusion pair may dominate the total error budget. Confidence may not reliably separate correct from incorrect predictions.
- **Diagnosis:** Class imbalance causes the model to default to majority-class predictions; the dominant confusion pair is the highest-leverage target for improvement.
- **Action:** Class-weighted or focal loss; oversampling of rare transitions; direct expensive XAI methods at the top confusion pair first.

> **Example — Helpdesk** (Notebook 10, 3,329 predictions, 65.4% overall): 0% accuracy on rare activities (Assign seriousness: 0/123). Top error: Take in charge ticket → Resolve ticket (537 errors).
>
> **Example — BPIC17** (Notebook 05, 100 samples, 78% correct): Confidence gap between correct (0.715) and incorrect (0.727) is not statistically significant (p=0.756). The model's confidence is actually slightly higher for incorrect predictions, meaning confidence alone cannot distinguish correct from incorrect.

---

## 8. Pathway Explorer

Trace specific process paths interactively and measure model accuracy per pathway.

### Signal 8.1 — Structural confusion at specific pathways **[Error]**

- **Observation:** The model consistently predicts the wrong successor at certain process pathways, regardless of other feature values. Some pathways are deterministic (100% one successor) yet the model still makes errors there.
- **Diagnosis:** Structural confusion — the model has learned the wrong default transition for this pathway. Errors on deterministic pathways are clear model failures.
- **Action:** Augment the misclassified transition heavily; add transition-specific bias; use the set of decision points as a natural partitioning for targeted error analysis.

> **Example — Helpdesk** (Notebook 21): Model consistently predicts wrong successor at certain pathways regardless of feature values.
>
> **Example — BPIC17** (Notebook 21, 6,301 cases, 155 DFG edges, 22 decision points): Key pathway A_Create Application → A_Submitted → W_Handle leads: 4,074 matching cases, 100% deterministic, model predicts correctly at p=0.644. The 22 decision points provide a natural partitioning for targeted error analysis.

---

## Summary

| Method | Signals | Categories |
|---|---|---|
| IG — Dataset Aggregate | 1.1, 1.2 | Data, Data |
| IG — Encoder-Decoder Split | 2.1, 2.2 | Arch, Arch |
| IG — Per-Value | 3.1 | Arch |
| IG — Per-Case / Comparative | 4.1, 4.2 | Error, Error |
| GA Counterfactual | 5.1, 5.2 | Data, Data |
| CART Surrogate | 6.1, 6.2, 6.3 | Error, Arch, Error |
| Confusion Matrix | 7.1 | Error |
| Pathway Explorer | 8.1 | Error |

**14 signals from 8 methods.** A practitioner can work through them in order: start with the confusion matrix (7.1) to identify where errors concentrate, then use IG and surrogates to diagnose why, and finally use counterfactuals to guide data augmentation.
