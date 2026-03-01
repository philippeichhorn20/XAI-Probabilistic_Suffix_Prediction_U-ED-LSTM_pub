"""
Explanation data structures for LORELEY.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from sklearn.tree import DecisionTreeClassifier


@dataclass
class LoreleyExplanation:
    """
    Container for LORELEY explanation results.

    Contains the factual rule (why the model made its prediction) and
    counterfactual rules (what changes would lead to different predictions).

    Attributes:
        instance: Encoded feature vector of the input prefix
        predicted_class: Index of the predicted class
        predicted_class_name: Name of the predicted class
        factual_rule: Rule describing path to the predicted class
        counterfactual_rules: List of rules for alternative predictions
        decision_tree: The trained decision tree surrogate model
        fidelity: Accuracy of the decision tree on held-out data
        feature_names: Names of all features in the encoded vector
        class_names: Names of all activity classes
    """

    instance: np.ndarray
    predicted_class: int
    predicted_class_name: str
    factual_rule: Dict[str, Any]
    counterfactual_rules: List[Dict[str, Any]]
    decision_tree: DecisionTreeClassifier
    fidelity: float
    feature_names: List[str]
    class_names: List[str]

    def __str__(self) -> str:
        """Human-readable representation of the explanation."""
        lines = [
            "LORELEY Explanation",
            "=" * 50,
            f"Predicted Class: {self.predicted_class_name} (idx={self.predicted_class})",
            f"Fidelity: {self.fidelity:.4f}",
            "",
            "Factual Rule:",
            self._format_rule(self.factual_rule),
            "",
            f"Counterfactual Rules ({len(self.counterfactual_rules)}):",
        ]
        for i, cf_rule in enumerate(self.counterfactual_rules[:5]):  # Show top 5
            lines.append(f"  [{i+1}] {self._format_rule(cf_rule)}")
        if len(self.counterfactual_rules) > 5:
            lines.append(f"  ... and {len(self.counterfactual_rules) - 5} more")
        return "\n".join(lines)

    def _format_rule(self, rule: Dict[str, Any]) -> str:
        """Format a rule dictionary as a readable string."""
        conditions = rule.get('conditions', [])
        outcome = rule.get('outcome', 'Unknown')
        outcome_name = rule.get('outcome_name', str(outcome))

        if not conditions:
            return f"  → {outcome_name}"

        cond_strs = []
        for cond in conditions:
            feature = cond['feature']
            threshold = cond['threshold']
            direction = cond['direction']
            cond_strs.append(f"{feature} {direction} {threshold:.3f}")

        return f"  IF {' AND '.join(cond_strs)} → {outcome_name}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to dictionary format."""
        return {
            'predicted_class': self.predicted_class,
            'predicted_class_name': self.predicted_class_name,
            'fidelity': self.fidelity,
            'factual_rule': self.factual_rule,
            'counterfactual_rules': self.counterfactual_rules,
            'n_counterfactuals': len(self.counterfactual_rules),
        }

    def get_simplest_counterfactual(self) -> Dict[str, Any]:
        """Get the counterfactual with fewest violated conditions."""
        if not self.counterfactual_rules:
            return None
        return self.counterfactual_rules[0]  # Already sorted by n_violations

    def get_counterfactuals_for_class(self, class_idx: int) -> List[Dict[str, Any]]:
        """Get all counterfactual rules leading to a specific class."""
        return [r for r in self.counterfactual_rules if r['outcome'] == class_idx]
