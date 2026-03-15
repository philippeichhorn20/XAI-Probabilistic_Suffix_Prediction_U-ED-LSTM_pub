"""
Declare constraint checking and mining for counterfactual generation.

Minimal implementation for REVISED+ - no external dependencies.
"""

from dataclasses import dataclass
from typing import List, Set, Optional, Tuple
from .templates import DeclareTemplate


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
