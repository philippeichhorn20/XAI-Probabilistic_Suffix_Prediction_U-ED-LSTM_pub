"""
Declare constraint checking and mining for counterfactual generation.

Minimal implementation for REVISED+ - no external dependencies.
Supports MP-Declare data conditions on activation events.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from .templates import DeclareTemplate


# =============================================================================
# MP-Declare Data Conditions
# =============================================================================

@dataclass(frozen=True)
class DataConjunct:
    """A single conjunct of an MP-Declare data condition.

    Represents ``feature_name operator value`` evaluated at an event position.
    For categorical features, comparison is ``cat_data[feature_idx][pos] == value_idx``.
    """
    feature_name: str
    feature_idx: int          # index into the categorical tensor list
    value_idx: int            # encoded integer value to compare against
    operator: str = "is"      # "is" (equality)


@dataclass(frozen=True)
class DataCondition:
    """Parsed activation condition for an MP-Declare constraint.

    A conjunction of :class:`DataConjunct` items.  The condition holds at a
    sequence position when **all** conjuncts are satisfied at that position.
    """
    conjuncts: Tuple[DataConjunct, ...]
    raw_string: str = ""      # original unparsed string, kept for display


@dataclass(frozen=True)
class DeclareConstraint:
    """A Declare constraint instance."""
    template: DeclareTemplate
    activities: Tuple[int, ...]
    parameter: Optional[int] = None

    def __str__(self) -> str:
        if self.template.is_unary():
            if self.template.requires_parameter():
                return f"{self.template.value}({self.activities[0]}, {self.parameter})"
            return f"{self.template.value}({self.activities[0]})"
        return f"{self.template.value}({self.activities[0]}, {self.activities[1]})"

    def format(self, activity_names: List[str]) -> str:
        """Format with human-readable activity names instead of indices."""
        def name(idx):
            if idx < len(activity_names):
                return activity_names[idx]
            return str(idx)

        if self.template.is_unary():
            if self.template.requires_parameter():
                return f"{self.template.value}({name(self.activities[0])}, n={self.parameter})"
            return f"{self.template.value}({name(self.activities[0])})"
        return f"{self.template.value}({name(self.activities[0])}, {name(self.activities[1])})"


class DeclareConstraintChecker:
    """Check if sequences satisfy Declare constraints."""

    @staticmethod
    def check(seq: List[int], c: DeclareConstraint) -> bool:
        """Check if sequence satisfies constraint."""
        t, acts, n = c.template, c.activities, c.parameter or 1

        # Unary
        if t == DeclareTemplate.INIT:
            return len(seq) > 0 and seq[0] == acts[0]
        if t == DeclareTemplate.LAST:
            return len(seq) > 0 and seq[-1] == acts[0]
        if t == DeclareTemplate.EXISTENCE:
            return seq.count(acts[0]) >= n
        if t == DeclareTemplate.ABSENCE:
            return seq.count(acts[0]) < n
        if t == DeclareTemplate.EXACTLY:
            return seq.count(acts[0]) == n

        # Binary
        a, b = acts
        if t == DeclareTemplate.CO_EXISTENCE:
            return (a in seq) == (b in seq)

        if t == DeclareTemplate.RESPONSE:
            for i, x in enumerate(seq):
                if x == a and b not in seq[i+1:]:
                    return False
            return True

        if t == DeclareTemplate.PRECEDENCE:
            seen_a = False
            for x in seq:
                if x == a: seen_a = True
                if x == b and not seen_a: return False
            return True

        if t == DeclareTemplate.SUCCESSION:
            return (DeclareConstraintChecker.check(seq, DeclareConstraint(DeclareTemplate.RESPONSE, acts)) and
                    DeclareConstraintChecker.check(seq, DeclareConstraint(DeclareTemplate.PRECEDENCE, acts)))

        if t == DeclareTemplate.NOT_SUCCESSION:
            for i, x in enumerate(seq):
                if x == a and b in seq[i+1:]:
                    return False
            return True

        if t == DeclareTemplate.ALTERNATE_RESPONSE:
            waiting = False
            for x in seq:
                if x == a:
                    if waiting: return False
                    waiting = True
                elif x == b:
                    waiting = False
            return not waiting

        if t == DeclareTemplate.ALTERNATE_PRECEDENCE:
            available = False
            for x in seq:
                if x == a: available = True
                elif x == b:
                    if not available: return False
                    available = False
            return True

        if t == DeclareTemplate.ALTERNATE_SUCCESSION:
            return (DeclareConstraintChecker.check(seq, DeclareConstraint(DeclareTemplate.ALTERNATE_RESPONSE, acts)) and
                    DeclareConstraintChecker.check(seq, DeclareConstraint(DeclareTemplate.ALTERNATE_PRECEDENCE, acts)))

        if t == DeclareTemplate.CHAIN_RESPONSE:
            for i, x in enumerate(seq[:-1]):
                if x == a and seq[i+1] != b:
                    return False
            return not (seq and seq[-1] == a)

        if t == DeclareTemplate.CHAIN_PRECEDENCE:
            for i, x in enumerate(seq):
                if x == b and (i == 0 or seq[i-1] != a):
                    return False
            return True

        if t == DeclareTemplate.CHAIN_SUCCESSION:
            return (DeclareConstraintChecker.check(seq, DeclareConstraint(DeclareTemplate.CHAIN_RESPONSE, acts)) and
                    DeclareConstraintChecker.check(seq, DeclareConstraint(DeclareTemplate.CHAIN_PRECEDENCE, acts)))

        return True

    @staticmethod
    def count_violations(seq: List[int], constraints: List[DeclareConstraint]) -> int:
        """Count violated constraints."""
        return sum(1 for c in constraints if not DeclareConstraintChecker.check(seq, c))

    @staticmethod
    def all_satisfied(seq: List[int], constraints: List[DeclareConstraint]) -> bool:
        """Check if all constraints satisfied."""
        return all(DeclareConstraintChecker.check(seq, c) for c in constraints)

    @staticmethod
    def satisfaction_rate(seq: List[int], constraints) -> float:
        """Fraction of constraints satisfied. Returns 1.0 if no constraints."""
        if not constraints:
            return 1.0
        n = len(constraints) if hasattr(constraints, '__len__') else sum(1 for _ in constraints)
        if n == 0:
            return 1.0
        sat = sum(1 for c in constraints if DeclareConstraintChecker.check(seq, c))
        return sat / n

    @staticmethod
    def prefix_safe_constraints(constraints) -> List[DeclareConstraint]:
        """Filter to only prefix-safe constraints (monotonic violations)."""
        return [c for c in constraints if c.template.is_prefix_safe()]

    # ------------------------------------------------------------------
    # MP-Declare: data-condition-aware checking
    # ------------------------------------------------------------------

    @staticmethod
    def _condition_holds(
        pos: int,
        condition: DataCondition,
        cat_data: Dict[int, List[int]],
    ) -> bool:
        """Return True if all conjuncts of *condition* are satisfied at *pos*."""
        for conj in condition.conjuncts:
            vals = cat_data.get(conj.feature_idx)
            if vals is None:
                # Feature not available — cannot verify, treat as not met
                return False
            if vals[pos] != conj.value_idx:
                return False
        return True

    @staticmethod
    def check_with_data(
        seq: List[int],
        c: DeclareConstraint,
        data_condition: Optional[DataCondition],
        cat_data: Optional[Dict[int, List[int]]] = None,
    ) -> bool:
        """Check constraint with an optional MP-Declare data condition.

        The *data_condition* restricts which occurrences of the activation
        activity trigger the constraint.  If ``None`` or *cat_data* is not
        provided, falls back to the standard :meth:`check`.

        Args:
            seq: Activity-index sequence (prefix, no padding).
            c: The Declare constraint.
            data_condition: Parsed activation condition, or ``None``.
            cat_data: ``{cat_feature_idx: [value_at_pos_0, …]}`` for each
                categorical feature available.  Lengths must equal ``len(seq)``.
        """
        if data_condition is None or not data_condition.conjuncts or cat_data is None:
            return DeclareConstraintChecker.check(seq, c)

        t, acts, n = c.template, c.activities, c.parameter or 1
        _holds = DeclareConstraintChecker._condition_holds

        # --- Unary templates ---
        if t == DeclareTemplate.INIT:
            if len(seq) == 0 or seq[0] != acts[0]:
                return False
            return _holds(0, data_condition, cat_data)

        if t == DeclareTemplate.LAST:
            if len(seq) == 0 or seq[-1] != acts[0]:
                return False
            return _holds(len(seq) - 1, data_condition, cat_data)

        if t == DeclareTemplate.EXISTENCE:
            cnt = sum(
                1 for i, x in enumerate(seq)
                if x == acts[0] and _holds(i, data_condition, cat_data)
            )
            return cnt >= n

        if t == DeclareTemplate.ABSENCE:
            cnt = sum(
                1 for i, x in enumerate(seq)
                if x == acts[0] and _holds(i, data_condition, cat_data)
            )
            return cnt < n

        if t == DeclareTemplate.EXACTLY:
            cnt = sum(
                1 for i, x in enumerate(seq)
                if x == acts[0] and _holds(i, data_condition, cat_data)
            )
            return cnt == n

        # --- Binary templates ---
        # Data condition filters activation (activities[0]) positions.
        a, b = acts

        def _is_active(pos: int) -> bool:
            """Position has the activation activity AND condition holds."""
            return seq[pos] == a and _holds(pos, data_condition, cat_data)

        if t == DeclareTemplate.CO_EXISTENCE:
            a_present = any(_is_active(i) for i in range(len(seq)))
            b_present = b in seq
            return a_present == b_present

        if t == DeclareTemplate.RESPONSE:
            for i in range(len(seq)):
                if _is_active(i) and b not in seq[i + 1:]:
                    return False
            return True

        if t == DeclareTemplate.PRECEDENCE:
            seen_a = False
            for i, x in enumerate(seq):
                if _is_active(i):
                    seen_a = True
                if x == b and not seen_a:
                    return False
            return True

        if t == DeclareTemplate.SUCCESSION:
            r = DeclareConstraintChecker.check_with_data(
                seq, DeclareConstraint(DeclareTemplate.RESPONSE, acts),
                data_condition, cat_data)
            p = DeclareConstraintChecker.check_with_data(
                seq, DeclareConstraint(DeclareTemplate.PRECEDENCE, acts),
                data_condition, cat_data)
            return r and p

        if t == DeclareTemplate.NOT_SUCCESSION:
            for i in range(len(seq)):
                if _is_active(i) and b in seq[i + 1:]:
                    return False
            return True

        if t == DeclareTemplate.ALTERNATE_RESPONSE:
            waiting = False
            for i, x in enumerate(seq):
                if _is_active(i):
                    if waiting:
                        return False
                    waiting = True
                elif x == b:
                    waiting = False
            return not waiting

        if t == DeclareTemplate.ALTERNATE_PRECEDENCE:
            available = False
            for i, x in enumerate(seq):
                if _is_active(i):
                    available = True
                elif x == b:
                    if not available:
                        return False
                    available = False
            return True

        if t == DeclareTemplate.ALTERNATE_SUCCESSION:
            r = DeclareConstraintChecker.check_with_data(
                seq, DeclareConstraint(DeclareTemplate.ALTERNATE_RESPONSE, acts),
                data_condition, cat_data)
            p = DeclareConstraintChecker.check_with_data(
                seq, DeclareConstraint(DeclareTemplate.ALTERNATE_PRECEDENCE, acts),
                data_condition, cat_data)
            return r and p

        if t == DeclareTemplate.CHAIN_RESPONSE:
            for i in range(len(seq) - 1):
                if _is_active(i) and seq[i + 1] != b:
                    return False
            if seq and _is_active(len(seq) - 1):
                return False
            return True

        if t == DeclareTemplate.CHAIN_PRECEDENCE:
            for i, x in enumerate(seq):
                if x == b and (i == 0 or not _is_active(i - 1)):
                    return False
            return True

        if t == DeclareTemplate.CHAIN_SUCCESSION:
            r = DeclareConstraintChecker.check_with_data(
                seq, DeclareConstraint(DeclareTemplate.CHAIN_RESPONSE, acts),
                data_condition, cat_data)
            p = DeclareConstraintChecker.check_with_data(
                seq, DeclareConstraint(DeclareTemplate.CHAIN_PRECEDENCE, acts),
                data_condition, cat_data)
            return r and p

        # Unknown template — fall back to standard check
        return DeclareConstraintChecker.check(seq, c)

    @staticmethod
    def satisfaction_rate_with_data(
        seq: List[int],
        constraints: List[DeclareConstraint],
        data_conditions: Optional[Dict[DeclareConstraint, DataCondition]] = None,
        cat_data: Optional[Dict[int, List[int]]] = None,
    ) -> float:
        """Fraction of constraints satisfied, using data conditions where available."""
        if not constraints:
            return 1.0
        n = len(constraints)
        dc_map = data_conditions or {}
        sat = sum(
            1 for c in constraints
            if DeclareConstraintChecker.check_with_data(
                seq, c, dc_map.get(c), cat_data)
        )
        return sat / n


class DeclareConstraintMiner:
    """Mine Declare constraints from sequences."""

    def __init__(self, vocab_size: int, min_support: float = 1.0):
        self.vocab_size = vocab_size
        self.min_support = min_support

    def mine(self, sequences: List[List[int]],
             templates: Optional[List[DeclareTemplate]] = None) -> Set[DeclareConstraint]:
        """Mine constraints with given support threshold."""
        if not sequences:
            return set()

        templates = templates or list(DeclareTemplate)
        n_seqs = len(sequences)
        min_count = int(n_seqs * self.min_support)
        constraints = set()

        # Get occurring activities
        activities = set()
        for seq in sequences:
            activities.update(seq)

        # Mine unary
        for a in activities:
            for t in templates:
                if not t.is_unary():
                    continue
                c = DeclareConstraint(t, (a,), 1 if t.requires_parameter() else None)
                if sum(1 for s in sequences if DeclareConstraintChecker.check(s, c)) >= min_count:
                    constraints.add(c)

        # Mine binary
        from tqdm.auto import tqdm
        activity_list = sorted(activities)
        for a in tqdm(activity_list, desc="Mining binary constraints", leave=False):
            for b in activity_list:
                if a == b:
                    continue
                for t in templates:
                    if not t.is_binary():
                        continue
                    c = DeclareConstraint(t, (a, b))
                    if sum(1 for s in sequences if DeclareConstraintChecker.check(s, c)) >= min_count:
                        constraints.add(c)

        return constraints

    def mine_label_specific(self, sequences: List[List[int]], labels: List[int],
                            target_label: int,
                            templates: Optional[List[DeclareTemplate]] = None) -> Set[DeclareConstraint]:
        """Mine constraints specific to target label (DLC)."""
        target_seqs = [s for s, l in zip(sequences, labels) if l == target_label]
        if len(target_seqs) < 5:
            return set()

        target_constraints = self.mine(target_seqs, templates)
        all_constraints = self.mine(sequences, templates)
        return target_constraints - all_constraints
