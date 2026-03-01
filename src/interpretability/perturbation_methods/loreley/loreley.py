"""
LORELEY: Counterfactual Explanations for Predictive Process Monitoring

Adapted from: Huang et al. (2022) - "Counterfactual Explanations for Predictive
Business Process Monitoring"

This implementation is adapted for next activity prediction.
See ADAPTATIONS.md for details on modifications from the original paper.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Any
from sklearn.tree import DecisionTreeClassifier
import random

# Optional dependency for edit distance
try:
    from Levenshtein import distance as levenshtein_distance
    _LEVENSHTEIN_AVAILABLE = True
except ImportError:
    _LEVENSHTEIN_AVAILABLE = False

    def levenshtein_distance(s1: str, s2: str) -> int:
        """Fallback Levenshtein distance implementation."""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

from .config import LoreleyConfig
from .encoder import PrefixEncoder
from .explanation import LoreleyExplanation
from .genetic import NeighborhoodGenerator

# Import base classes
from ..base import ProcessModelPredictor


class LORELEY:
    """
    LORELEY: LOcal Rule-based Explanations for predictive process monitoring.

    Generates counterfactual explanations by:
    1. Finding similar prefixes using Levenshtein edit distance
    2. Generating synthetic neighborhood via genetic algorithm
    3. Training interpretable decision tree surrogate
    4. Extracting factual and counterfactual rules

    Example:
        >>> loreley = create_loreley_for_model(model, categories, num_features)
        >>> explanation = loreley.explain(prefix, training_prefixes)
        >>> print(explanation)
    """

    def __init__(self,
                 model,
                 encoder: PrefixEncoder,
                 activity_names: List[str],
                 config: Optional[LoreleyConfig] = None):
        """
        Args:
            model: The prediction model with predict(prefix) -> class_idx method
            encoder: PrefixEncoder for converting prefixes to feature vectors
            activity_names: List of activity names (classes)
            config: LORELEY configuration
        """
        self.model = model
        self.encoder = encoder
        self.activity_names = activity_names
        self.n_classes = len(activity_names)
        self.config = config or LoreleyConfig()

        # Initialize genetic neighborhood generator
        self._neighborhood_generator = NeighborhoodGenerator(
            population_size=self.config.population_size,
            n_generations=self.config.n_generations,
            crossover_prob=self.config.crossover_prob,
            mutation_prob=self.config.mutation_prob,
            seed=self.config.seed
        )

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def explain(self,
                prefix: Tuple[List[torch.Tensor], List[torch.Tensor]],
                training_prefixes: List[Tuple[List[torch.Tensor], List[torch.Tensor]]],
                target_class: Optional[int] = None) -> LoreleyExplanation:
        """
        Generate counterfactual explanation for a prefix.

        Args:
            prefix: The prefix to explain (cat_tensors, num_tensors)
            training_prefixes: List of training prefixes for finding similar instances
            target_class: If None, explain the predicted class

        Returns:
            LoreleyExplanation with factual and counterfactual rules
        """
        # Encode the instance
        x = self.encoder.encode(prefix)

        # Get prediction
        predicted_class = self._predict(prefix)
        if target_class is None:
            target_class = predicted_class

        # Stage 0: Find similar prefixes using edit distance
        similar_prefixes = self._find_similar_prefixes(prefix, training_prefixes)

        # Stage 1: Generate neighborhood using genetic algorithm
        Z, Y = self._generate_neighborhood(x, similar_prefixes, target_class)

        # Stage 2: Train decision tree and extract rules
        tree, fidelity = self._train_decision_tree(Z, Y, x, target_class)

        # Extract rules
        factual_rule = self._extract_factual_rule(tree, x, target_class)
        counterfactual_rules = self._extract_counterfactual_rules(tree, x, target_class)

        return LoreleyExplanation(
            instance=x,
            predicted_class=predicted_class,
            predicted_class_name=self.activity_names[predicted_class],
            factual_rule=factual_rule,
            counterfactual_rules=counterfactual_rules,
            decision_tree=tree,
            fidelity=fidelity,
            feature_names=self.encoder.feature_names,
            class_names=self.activity_names
        )

    def _predict(self, prefix: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> int:
        """Get model prediction for a prefix."""
        return self.model.predict(prefix)

    def _get_activity_sequence(self,
                                prefix: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> List[int]:
        """Extract activity sequence from prefix."""
        cat_tensors, _ = prefix
        return cat_tensors[0][0].cpu().numpy().tolist()

    def _find_similar_prefixes(self,
                                prefix: Tuple[List[torch.Tensor], List[torch.Tensor]],
                                training_prefixes: List[Tuple],
                                max_similar: int = 100) -> List[Tuple]:
        """
        Find similar prefixes using Levenshtein edit distance on activity sequences.

        Args:
            prefix: The instance to explain
            training_prefixes: Pool of training prefixes
            max_similar: Maximum number of similar prefixes to return

        Returns:
            List of similar prefixes sorted by edit distance
        """
        target_seq = self._get_activity_sequence(prefix)
        target_str = ''.join(map(str, target_seq))

        similarities = []
        for train_prefix in training_prefixes:
            train_seq = self._get_activity_sequence(train_prefix)
            train_str = ''.join(map(str, train_seq))

            dist = levenshtein_distance(target_str, train_str)
            if dist <= self.config.edit_distance_threshold:
                similarities.append((dist, train_prefix))

        # Sort by distance and return top-k
        similarities.sort(key=lambda x: x[0])
        return [p for _, p in similarities[:max_similar]]

    def _generate_neighborhood(self,
                                x: np.ndarray,
                                similar_prefixes: List[Tuple],
                                target_class: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic neighborhood using genetic algorithm.

        Args:
            x: Encoded instance to explain
            similar_prefixes: Similar prefixes for initialization
            target_class: The predicted class

        Returns:
            Z: Synthetic instances [n_samples, n_features]
            Y: Class labels [n_samples]
        """
        # Encode similar prefixes for the initial population
        if similar_prefixes:
            init_population = np.array([self.encoder.encode(p) for p in similar_prefixes])
        else:
            # Fallback: duplicate x
            init_population = np.tile(x, (self.config.population_size // 2, 1))

        control_flow_indices = self.encoder.get_control_flow_indices()

        # Generate neighborhood for each class
        all_Z = []
        all_Y = []

        for class_idx in range(self.n_classes):
            Z_class = self._neighborhood_generator.evolve(
                x=x,
                init_population=init_population,
                target_class_idx=class_idx,
                control_flow_indices=control_flow_indices
            )

            Y_class = np.full(len(Z_class), class_idx)

            all_Z.append(Z_class)
            all_Y.append(Y_class)

        Z = np.vstack(all_Z)
        Y = np.concatenate(all_Y)

        return Z, Y

    def _train_decision_tree(self,
                             Z: np.ndarray,
                             Y: np.ndarray,
                             x: np.ndarray,
                             target_class: int) -> Tuple[DecisionTreeClassifier, float]:
        """
        Train decision tree on synthetic neighborhood.

        Returns:
            tree: Trained DecisionTreeClassifier
            fidelity: Accuracy of tree on held-out synthetic data
        """
        # Split into train/test
        n_test = int(len(Z) * self.config.test_ratio)
        indices = np.random.permutation(len(Z))
        train_idx, test_idx = indices[n_test:], indices[:n_test]

        Z_train, Y_train = Z[train_idx], Y[train_idx]
        Z_test, Y_test = Z[test_idx], Y[test_idx]

        # Train decision tree
        tree = DecisionTreeClassifier(
            max_depth=self.config.max_tree_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.seed
        )
        tree.fit(Z_train, Y_train)

        # Calculate fidelity
        if len(Z_test) > 0:
            fidelity = tree.score(Z_test, Y_test)
        else:
            fidelity = tree.score(Z_train, Y_train)

        return tree, fidelity

    def _extract_factual_rule(self,
                              tree: DecisionTreeClassifier,
                              x: np.ndarray,
                              target_class: int) -> Dict[str, Any]:
        """Extract the factual rule (path to x's leaf)."""
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        node = 0  # Start at root
        conditions = []

        while feature[node] != -2:  # -2 indicates leaf
            feat_idx = feature[node]
            thresh = threshold[node]
            feat_name = self.encoder.feature_names[feat_idx]

            if x[feat_idx] <= thresh:
                conditions.append({
                    'feature': feat_name,
                    'feature_idx': feat_idx,
                    'threshold': thresh,
                    'direction': '≤'
                })
                node = tree.tree_.children_left[node]
            else:
                conditions.append({
                    'feature': feat_name,
                    'feature_idx': feat_idx,
                    'threshold': thresh,
                    'direction': '>'
                })
                node = tree.tree_.children_right[node]

        # Get predicted class at leaf
        leaf_class = np.argmax(tree.tree_.value[node])

        return {
            'conditions': conditions,
            'outcome': int(leaf_class),
            'outcome_name': self.activity_names[leaf_class] if leaf_class < len(self.activity_names) else str(leaf_class)
        }

    def _extract_counterfactual_rules(self,
                                       tree: DecisionTreeClassifier,
                                       x: np.ndarray,
                                       target_class: int) -> List[Dict[str, Any]]:
        """
        Extract counterfactual rules (paths to leaves with different outcomes).

        Returns rules sorted by number of violated conditions (fewer = simpler).
        """
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        value = tree.tree_.value

        def get_leaf_rules(node: int, conditions: List[Dict]) -> List[Dict]:
            """Recursively extract rules for all leaves."""
            if feature[node] == -2:  # Leaf
                leaf_class = int(np.argmax(value[node]))
                return [{
                    'conditions': conditions.copy(),
                    'outcome': leaf_class,
                    'outcome_name': self.activity_names[leaf_class] if leaf_class < len(self.activity_names) else str(leaf_class)
                }]

            feat_idx = feature[node]
            thresh = threshold[node]
            feat_name = self.encoder.feature_names[feat_idx]

            # Left branch
            left_cond = conditions + [{
                'feature': feat_name,
                'feature_idx': feat_idx,
                'threshold': thresh,
                'direction': '≤'
            }]
            left_rules = get_leaf_rules(tree.tree_.children_left[node], left_cond)

            # Right branch
            right_cond = conditions + [{
                'feature': feat_name,
                'feature_idx': feat_idx,
                'threshold': thresh,
                'direction': '>'
            }]
            right_rules = get_leaf_rules(tree.tree_.children_right[node], right_cond)

            return left_rules + right_rules

        all_rules = get_leaf_rules(0, [])

        # Filter for different outcomes
        counterfactual_rules = [r for r in all_rules if r['outcome'] != target_class]

        # Count violated conditions for each rule
        for rule in counterfactual_rules:
            violations = 0
            for cond in rule['conditions']:
                feat_idx = cond['feature_idx']
                thresh = cond['threshold']
                direction = cond['direction']

                if direction == '≤' and x[feat_idx] > thresh:
                    violations += 1
                elif direction == '>' and x[feat_idx] <= thresh:
                    violations += 1

            rule['n_violations'] = violations

        # Sort by number of violations
        counterfactual_rules.sort(key=lambda r: r['n_violations'])

        return counterfactual_rules


class LoreleyModelPredictor:
    """
    Wrapper to provide predict(prefix) -> class_idx interface for LORELEY.

    Handles the specific input format expected by U-ED-LSTM models.
    """

    def __init__(self, model, suffix_step: int = 0):
        """
        Args:
            model: The U-ED-LSTM model
            suffix_step: Which suffix step to predict (0 = next activity)
        """
        self.model = model
        self.suffix_step = suffix_step
        self.device = next(model.parameters()).device

    def predict(self, prefix: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> int:
        """Predict the next activity class index."""
        cat_tensors, num_tensors = prefix

        # Move to device
        cat_tensors = [t.to(self.device) for t in cat_tensors]
        num_tensors = [t.to(self.device) for t in num_tensors]

        with torch.no_grad():
            cat_preds, num_preds = self.model((cat_tensors, num_tensors))
            activity_logits = cat_preds["Activity_mean"][self.suffix_step]
            predicted_class = activity_logits.argmax(dim=-1).item()

        return predicted_class

    def predict_proba(self, prefix: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> np.ndarray:
        """Predict class probabilities."""
        cat_tensors, num_tensors = prefix

        cat_tensors = [t.to(self.device) for t in cat_tensors]
        num_tensors = [t.to(self.device) for t in num_tensors]

        with torch.no_grad():
            cat_preds, num_preds = self.model((cat_tensors, num_tensors))
            activity_logits = cat_preds["Activity_mean"][self.suffix_step]
            probs = torch.softmax(activity_logits, dim=-1)

        return probs.cpu().numpy()[0]


def create_loreley_for_model(model,
                              data_set_categories: Dict,
                              num_feature_names: List[str],
                              config: Optional[LoreleyConfig] = None) -> LORELEY:
    """
    Factory function to create LORELEY for a given model.

    Args:
        model: U-ED-LSTM model
        data_set_categories: Dictionary mapping feature names to vocabularies
        num_feature_names: List of numerical feature names
        config: LORELEY configuration

    Returns:
        Configured LORELEY instance
    """
    # Get activity vocabulary
    activity_vocab = data_set_categories.get('Activity', [])
    activity_vocab_size = len(activity_vocab)
    activity_names = list(activity_vocab)

    # Get other categorical features
    cat_feature_names = [k for k in data_set_categories.keys() if k != 'Activity']
    cat_vocab_sizes = [len(data_set_categories[k]) for k in cat_feature_names]

    # Create encoder
    encoder = PrefixEncoder(
        activity_vocab_size=activity_vocab_size,
        num_feature_names=num_feature_names,
        cat_feature_names=cat_feature_names,
        cat_vocab_sizes=cat_vocab_sizes
    )

    # Create model predictor wrapper
    predictor = LoreleyModelPredictor(model, suffix_step=0)

    return LORELEY(
        model=predictor,
        encoder=encoder,
        activity_names=activity_names,
        config=config
    )
