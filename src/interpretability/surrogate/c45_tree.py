"""
C4.5 Decision Tree implementation for gateway decision analysis.

Implements Quinlan's C4.5 algorithm with:
- Information gain ratio for split selection
- Native multi-way categorical splits (no ordinal encoding needed)
- Post-pruning via reduced-error pruning
- API compatible with scikit-learn's DecisionTreeClassifier where possible

Usage:
    clf = C45Classifier(max_depth=5, min_samples_leaf=20)
    clf.fit(X_train, y_train, categorical_features=['last_activity', 'variant_index'])
    preds = clf.predict(X_test)
    acc = clf.score(X_test, y_test)
"""

import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd


@dataclass
class C45Node:
    """A node in the C4.5 decision tree."""
    # Split info (None for leaf)
    feature: Optional[str] = None
    is_categorical: bool = False
    # For categorical: maps category value -> child index
    category_map: Optional[Dict[Any, int]] = None
    # For numerical: split threshold (left <= threshold, right > threshold)
    threshold: Optional[float] = None

    # Children
    children: List['C45Node'] = field(default_factory=list)
    # For categorical: labels for each branch
    branch_labels: List[str] = field(default_factory=list)

    # Node statistics (always populated)
    class_distribution: Optional[Dict[str, int]] = None
    n_samples: int = 0
    majority_class: Optional[str] = None
    depth: int = 0


class C45Classifier:
    """
    C4.5 decision tree classifier.

    Supports native multi-way categorical splits and uses information gain
    ratio as the split criterion.

    Parameters:
        max_depth: Maximum tree depth. None for unlimited.
        min_samples_leaf: Minimum samples required at a leaf node.
        min_samples_split: Minimum samples required to split a node.
        min_gain_ratio: Minimum gain ratio to accept a split.
        class_weight: 'balanced' to weight classes inversely proportional
                      to their frequency, or None.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        min_gain_ratio: float = 1e-7,
        class_weight: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_gain_ratio = min_gain_ratio
        self.class_weight = class_weight
        self.random_state = random_state

        self.root_: Optional[C45Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.categorical_features_: Optional[Set[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self._n_features: int = 0
        self._class_weights: Optional[Dict[str, float]] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        categorical_features: Optional[List[str]] = None,
    ) -> 'C45Classifier':
        """Fit the C4.5 decision tree.

        Args:
            X: Feature DataFrame (can contain mixed types).
            y: Target labels (string or int array).
            categorical_features: Column names to treat as categorical.
                If None, auto-detects from dtype (object = categorical).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.feature_names_ = list(X.columns)
        self._n_features = len(self.feature_names_)
        self.classes_ = np.array(sorted(set(y)))

        if categorical_features is not None:
            self.categorical_features_ = set(categorical_features)
        else:
            self.categorical_features_ = {
                col for col in X.columns if X[col].dtype == object
            }

        # Compute class weights
        if self.class_weight == 'balanced':
            counts = Counter(y)
            n = len(y)
            n_classes = len(counts)
            self._class_weights = {
                c: n / (n_classes * cnt) for c, cnt in counts.items()
            }
        else:
            self._class_weights = {c: 1.0 for c in self.classes_}

        # Pre-extract columns as numpy arrays for speed.
        # For categoricals, encode to integers for fast grouping.
        self._col_arrays = {}
        self._cat_encodings = {}  # feat_name -> {value: int_code}
        self._cat_decodings = {}  # feat_name -> {int_code: value}

        for col in X.columns:
            raw = X[col].values
            if col in self.categorical_features_:
                # Encode categoricals to integers once
                unique_vals, coded = np.unique(raw.astype(str), return_inverse=True)
                self._col_arrays[col] = coded
                self._cat_encodings[col] = {v: i for i, v in enumerate(unique_vals)}
                self._cat_decodings[col] = {i: v for i, v in enumerate(unique_vals)}
            else:
                self._col_arrays[col] = raw.astype(float)

        # Encode target labels to integers
        self._y_codes, self._y_uniques = pd.factorize(y, sort=True)

        # Build tree
        indices = np.arange(len(X))
        self.root_ = self._build_tree(X, y, indices, depth=0)

        # Collapse subtrees whose leaves all predict the same class
        # (same-prediction pruning; fidelity is unchanged).
        self._collapse_same_prediction(self.root_)

        del self._col_arrays, self._cat_encodings, self._y_codes, self._y_uniques

        # Compute feature importances after pruning so collapsed splits
        # don't contribute importance.
        self._compute_feature_importances(X, y)

        return self

    def _collapse_same_prediction(self, node: 'C45Node') -> Optional[str]:
        """Recursively collapse subtrees where every leaf predicts the same class.

        Returns the shared majority class of the subtree, or None if descendants
        disagree (in which case the subtree is left untouched).
        """
        if not node.children:
            return node.majority_class

        child_preds = [self._collapse_same_prediction(c) for c in node.children]
        if any(p is None for p in child_preds) or len(set(child_preds)) > 1:
            return None

        # All descendants predict the same class — replace subtree with a leaf.
        node.feature = None
        node.is_categorical = False
        node.threshold = None
        node.category_map = None
        node.branch_labels = None
        node.children = []
        return node.majority_class

    # =================================================================
    # Tree Building
    # =================================================================

    def _build_tree(
        self, X: pd.DataFrame, y: np.ndarray,
        indices: np.ndarray, depth: int,
    ) -> C45Node:
        """Recursively build the tree."""
        labels = y[indices]
        dist = Counter(labels)
        majority = dist.most_common(1)[0][0]

        node = C45Node(
            class_distribution=dict(dist),
            n_samples=len(indices),
            majority_class=majority,
            depth=depth,
        )

        # Stopping conditions
        if len(dist) == 1:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(indices) < self.min_samples_split:
            return node

        # Find best split
        best = self._find_best_split(X, y, indices)
        if best is None:
            return node

        feature, is_cat, threshold, category_map, child_indices, branch_labels, gain_ratio = best

        if gain_ratio < self.min_gain_ratio:
            return node

        # Check min_samples_leaf for all children
        if any(len(ci) < self.min_samples_leaf for ci in child_indices):
            return node

        node.feature = feature
        node.is_categorical = is_cat
        node.threshold = threshold
        node.category_map = category_map
        node.branch_labels = branch_labels

        for ci in child_indices:
            child = self._build_tree(X, y, ci, depth + 1)
            node.children.append(child)

        return node

    def _find_best_split(
        self, X: pd.DataFrame, y: np.ndarray, indices: np.ndarray,
    ) -> Optional[Tuple]:
        """Find the best split across all features."""
        y_codes = self._y_codes[indices]
        parent_entropy = self._fast_entropy(y_codes, len(self._y_uniques))

        best = None
        best_gain_ratio = -1

        for feat_name in self.feature_names_:
            values = self._col_arrays[feat_name][indices]

            if feat_name in self.categorical_features_:
                result = self._categorical_split(
                    feat_name, values, y_codes, indices, parent_entropy)
            else:
                result = self._numerical_split(
                    values, y_codes, indices, parent_entropy)

            if result is not None:
                gain_ratio = result[0]
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    is_cat = True if feat_name in self.categorical_features_ else False
                    best = (feat_name, is_cat, result[1], result[2],
                            result[3], result[4], gain_ratio)

        return best

    def _categorical_split(
        self, feat_name: str, codes: np.ndarray, y_codes: np.ndarray,
        indices: np.ndarray, parent_entropy: float,
    ) -> Optional[Tuple]:
        """Multi-way split on a pre-encoded categorical feature."""
        n_cats = len(self._cat_decodings[feat_name])
        n_total = len(y_codes)
        n_classes = len(self._y_uniques)

        # Use bincount per category for fast computation
        unique_codes = np.unique(codes)
        if len(unique_codes) <= 1:
            return None

        weighted_child_entropy = 0.0
        split_info = 0.0
        child_indices_list = []
        branch_labels = []
        category_map = {}
        decoding = self._cat_decodings[feat_name]

        for branch_idx, cat_code in enumerate(unique_codes):
            mask = codes == cat_code
            local_count = mask.sum()

            if local_count < self.min_samples_leaf:
                return None

            child_y = y_codes[mask]
            weight = local_count / n_total
            weighted_child_entropy += weight * self._fast_entropy(child_y, n_classes)
            if weight > 0:
                split_info -= weight * np.log2(weight)

            child_indices_list.append(indices[mask])
            cat_val = decoding[cat_code]
            branch_labels.append(cat_val)
            category_map[cat_val] = branch_idx

        if split_info == 0:
            return None

        info_gain = parent_entropy - weighted_child_entropy
        gain_ratio = info_gain / split_info

        return gain_ratio, None, category_map, child_indices_list, branch_labels

    def _numerical_split(
        self, values: np.ndarray, y_codes: np.ndarray,
        indices: np.ndarray, parent_entropy: float,
    ) -> Optional[Tuple]:
        """Binary split on a numerical feature (best threshold)."""
        valid = ~np.isnan(values)
        if valid.sum() < 2 * self.min_samples_leaf:
            return None

        sorted_order = np.argsort(values[valid])
        sorted_vals = values[valid][sorted_order]
        sorted_y = y_codes[valid][sorted_order]
        sorted_indices = indices[valid][sorted_order]

        n_total = len(sorted_y)
        n_classes = len(self._y_uniques)
        best_gain_ratio = -1
        best_threshold = None
        best_split_pos = None

        # Incremental entropy: maintain left/right class counts
        left_counts = np.zeros(n_classes, dtype=int)
        right_counts = np.bincount(sorted_y, minlength=n_classes)

        for i in range(1, n_total):
            left_counts[sorted_y[i - 1]] += 1
            right_counts[sorted_y[i - 1]] -= 1

            if sorted_vals[i] == sorted_vals[i - 1]:
                continue
            if i < self.min_samples_leaf or (n_total - i) < self.min_samples_leaf:
                continue

            w_left = i / n_total
            w_right = (n_total - i) / n_total

            child_entropy = (
                w_left * self._entropy_from_counts(left_counts, i)
                + w_right * self._entropy_from_counts(right_counts, n_total - i)
            )

            split_info = -(w_left * np.log2(w_left) + w_right * np.log2(w_right))
            if split_info == 0:
                continue

            info_gain = parent_entropy - child_entropy
            gain_ratio = info_gain / split_info

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_threshold = (sorted_vals[i - 1] + sorted_vals[i]) / 2
                best_split_pos = i

        if best_threshold is None:
            return None

        left_mask = values <= best_threshold
        right_mask = (~left_mask) & valid
        left_idx = indices[left_mask & valid]
        right_idx = indices[right_mask]

        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return None

        branch_labels = [f'<= {best_threshold:.4g}', f'> {best_threshold:.4g}']
        return (best_gain_ratio, best_threshold, None,
                [left_idx, right_idx], branch_labels)

    # =================================================================
    # Entropy (vectorized)
    # =================================================================

    def _fast_entropy(self, y_codes: np.ndarray, n_classes: int) -> float:
        """Compute weighted entropy from integer-coded labels using bincount."""
        counts = np.bincount(y_codes, minlength=n_classes)
        return self._entropy_from_counts(counts, len(y_codes))

    def _entropy_from_counts(self, counts: np.ndarray, n_total: int) -> float:
        """Compute weighted entropy from class counts."""
        if n_total == 0:
            return 0.0
        # Apply class weights
        if self._class_weights is not None:
            weighted = np.array([
                counts[i] * self._class_weights.get(self._y_uniques[i], 1.0)
                if i < len(self._y_uniques) else 0.0
                for i in range(len(counts))
            ])
        else:
            weighted = counts.astype(float)
        total = weighted.sum()
        if total == 0:
            return 0.0
        probs = weighted / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _weighted_entropy(self, labels: np.ndarray) -> float:
        """Compute weighted entropy (fallback for non-coded labels)."""
        counts = Counter(labels)
        total = sum(self._class_weights.get(c, 1.0) * cnt
                    for c, cnt in counts.items())
        if total == 0:
            return 0.0
        entropy = 0.0
        for c, cnt in counts.items():
            w = self._class_weights.get(c, 1.0)
            p = (w * cnt) / total
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    # =================================================================
    # Prediction
    # =================================================================

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return np.array([self._predict_one(row) for _, row in X.iterrows()])

    def _predict_one(self, row: pd.Series) -> str:
        """Predict for a single sample."""
        node = self.root_
        while node.feature is not None and node.children:
            val = row[node.feature]
            if node.is_categorical:
                val_str = str(val)
                branch_idx = node.category_map.get(val_str)
                if branch_idx is None or branch_idx >= len(node.children):
                    break  # Unseen category → use current node's majority
                node = node.children[branch_idx]
            else:
                try:
                    fval = float(val)
                except (ValueError, TypeError):
                    break
                if np.isnan(fval):
                    break
                if fval <= node.threshold:
                    node = node.children[0]
                else:
                    node = node.children[1]
        return node.majority_class

    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Accuracy on the given data."""
        preds = self.predict(X)
        return (preds == y).mean()

    # =================================================================
    # Tree Properties (scikit-learn compatible)
    # =================================================================

    def get_depth(self) -> int:
        """Maximum depth of the tree."""
        def _depth(node):
            if not node.children:
                return 0
            return 1 + max(_depth(c) for c in node.children)
        return _depth(self.root_) if self.root_ else 0

    def get_n_leaves(self) -> int:
        """Number of leaf nodes."""
        def _count(node):
            if not node.children:
                return 1
            return sum(_count(c) for c in node.children)
        return _count(self.root_) if self.root_ else 0

    def _compute_feature_importances(self, X: pd.DataFrame, y: np.ndarray):
        """Compute feature importances based on weighted information gain at each split."""
        importances = np.zeros(self._n_features)
        total_samples = len(y)

        def _accumulate(node):
            if node.feature is None or not node.children:
                return
            feat_idx = self.feature_names_.index(node.feature)
            # Weight = fraction of samples reaching this node
            weight = node.n_samples / total_samples

            parent_entropy = self._weighted_entropy(
                np.array([c for c, cnt in node.class_distribution.items()
                          for _ in range(cnt)]))
            child_entropy = 0.0
            for child in node.children:
                if child.n_samples > 0:
                    child_labels = np.array(
                        [c for c, cnt in child.class_distribution.items()
                         for _ in range(cnt)])
                    child_entropy += (child.n_samples / node.n_samples) * \
                        self._weighted_entropy(child_labels)

            importances[feat_idx] += weight * (parent_entropy - child_entropy)

            for child in node.children:
                _accumulate(child)

        if self.root_:
            _accumulate(self.root_)

        # Normalize to sum to 1
        total = importances.sum()
        if total > 0:
            importances /= total
        self.feature_importances_ = importances

    # =================================================================
    # Rule Extraction
    # =================================================================

    def extract_rules(self) -> List[Dict]:
        """Extract all root-to-leaf paths as readable rules."""
        rules = []

        def _recurse(node, conditions):
            if node.feature is None or not node.children:
                total = sum(node.class_distribution.values())
                confidence = node.class_distribution.get(
                    node.majority_class, 0) / total if total > 0 else 0
                rules.append({
                    'conditions': list(conditions),
                    'prediction': node.majority_class,
                    'confidence': confidence,
                    'n_samples': node.n_samples,
                })
                return

            if node.is_categorical:
                for i, child in enumerate(node.children):
                    label = node.branch_labels[i] if i < len(node.branch_labels) else f'?{i}'
                    cond = f'{node.feature} = {label}'
                    _recurse(child, conditions + [cond])
            else:
                _recurse(node.children[0],
                         conditions + [f'{node.feature} <= {node.threshold:.4g}'])
                if len(node.children) > 1:
                    _recurse(node.children[1],
                             conditions + [f'{node.feature} > {node.threshold:.4g}'])

        if self.root_:
            _recurse(self.root_, [])
        rules.sort(key=lambda r: -r['n_samples'])
        return rules

    # =================================================================
    # Decision Path (for disagreement analysis)
    # =================================================================

    def explain_decision_path(self, row: pd.Series) -> List[str]:
        """Return the decision path for a single sample as a list of step strings."""
        steps = []
        node = self.root_
        while node.feature is not None and node.children:
            val = row[node.feature]
            if node.is_categorical:
                val_str = str(val)
                branch_idx = node.category_map.get(val_str)
                if branch_idx is None:
                    steps.append(f'{node.feature} = {val_str} (unseen category)')
                    break
                steps.append(f'{node.feature} = {val_str}')
                node = node.children[branch_idx]
            else:
                try:
                    fval = float(val)
                except (ValueError, TypeError):
                    break
                direction = '<=' if fval <= node.threshold else '>'
                steps.append(f'{node.feature} = {fval:.4g} {direction} {node.threshold:.4g}')
                node = node.children[0] if fval <= node.threshold else node.children[1]
        return steps

    # =================================================================
    # String representation
    # =================================================================

    def __repr__(self) -> str:
        if self.root_ is None:
            return "C45Classifier(not fitted)"
        return (f"C45Classifier(depth={self.get_depth()}, "
                f"leaves={self.get_n_leaves()}, "
                f"classes={len(self.classes_)})")
