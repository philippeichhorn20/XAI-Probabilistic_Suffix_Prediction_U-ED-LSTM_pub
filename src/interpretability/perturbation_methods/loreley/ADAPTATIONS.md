# LORELEY Adaptations for Next Activity Prediction

## Original Paper
**Huang et al. (2022)** - "Counterfactual Explanations for Predictive Business Process Monitoring"

## Original Use Case
- **Task**: Process outcome prediction (e.g., A_Pending, A_Canceled, A_Denied)
- **Model**: LSTM for multi-class classification
- **Output**: Binary/multi-class outcome labels

## Our Use Case
- **Task**: Next activity prediction
- **Model**: U-ED-LSTM (Encoder-Decoder LSTM with uncertainty)
- **Output**: Probability distribution over activities

---

## Adaptations Made

### 1. Target Prediction Type
| Original | Adaptation |
|----------|------------|
| Outcome prediction (3 classes) | Next activity prediction (N activities) |
| Classes: A_Pending, A_Canceled, A_Denied | Classes: All possible activities in the process |

**Rationale**: The genetic algorithm and decision tree can handle multi-class, but with many activity classes (e.g., 20+), the decision tree may become complex.

**Implementation Note**: We use `suffix_step=0` to get next activity predictions from the decoder.

### 2. Prefix Encoding
| Original | Adaptation |
|----------|------------|
| Index-based or aggregation encoding | Raw sequential tensors (cat_tensors, num_tensors) |
| Fixed-length feature vectors | Variable-length prefix sequences |

**Rationale**: Our model uses embeddings directly, not aggregated features. For the genetic algorithm and decision tree, we convert prefixes to a fixed-length representation using:
- Activity frequency counts (control flow)
- Summary statistics for numerical features (mean, sum, max)
- One-hot encoded case attributes

### 3. Edit Distance Computation
| Original | Adaptation |
|----------|------------|
| Levenshtein distance on activity sequences | Levenshtein distance on activity index sequences |

**Implementation**: We extract the activity sequence from the prefix and compute edit distance directly.

### 4. Control Flow Representation
| Original | Adaptation |
|----------|------------|
| Frequency vector of activities | Frequency vector of activities |

**Note**: Same as original - we count occurrences of each activity in the prefix.

### 5. Fitness Function
| Original | Adaptation |
|----------|------------|
| `fitness_i(z) = 1_{b(z)=i} + (1 - d(x,z)) - 1_{x=z}` | Same formula |
| Distance d on encoded vectors | Distance d on frequency-encoded vectors |

**Note**: The fitness function remains the same, but distance is computed on our encoding.

### 6. Black-Box Model Interface
| Original | Adaptation |
|----------|------------|
| Direct model prediction | Wrapper through IGModelWrapper |
| `model.predict(x)` | `get_next_activity_prediction(prefix)` |

**Implementation**: We wrap our model to provide a simple `predict(encoded_prefix) -> class_idx` interface.

### 7. Decision Tree Interpretable Model
| Original | Adaptation |
|----------|------------|
| sklearn DecisionTreeClassifier | sklearn DecisionTreeClassifier |
| Trained on synthetic neighborhood | Same |

**Note**: No change needed - decision trees work with any number of classes.

---

## Limitations and Future Work

### Current Limitations
1. **Many Activity Classes**: With 20+ activities, counterfactual rules may become complex
2. **Sequential Information Loss**: Frequency encoding loses ordering information
3. **Numerical Feature Handling**: Summary statistics may not capture temporal patterns

### Potential Improvements
1. Use sequence embeddings for similarity computation
2. Add temporal constraints to the genetic algorithm
3. Implement prefix-length-aware bucketing

---

## API Differences

### Original LORELEY
```python
loreley = LORELEY(black_box_model, event_log)
explanation = loreley.explain(instance_to_explain)
```

### Our Adaptation
```python
loreley = LORELEY(model_wrapper, config)
explanation = loreley.explain(
    prefix,           # (cat_tensors, num_tensors)
    event_log,        # Training data for finding similar prefixes
    target_class=None # None = explain predicted class
)
# Returns: LoreleyExplanation with factual_rule, counterfactual_rules
```

---

## Validation Checklist
- [ ] Edit distance finds similar prefixes correctly
- [ ] Genetic algorithm generates valid synthetic prefixes
- [ ] Control flow constraints are respected during crossover/mutation
- [ ] Decision tree fidelity is measured and reported
- [ ] Counterfactual rules are actionable (realistic changes)
