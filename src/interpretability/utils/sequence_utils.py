"""
Sequence formatting and display utilities for interpretability.

Provides reusable functions for displaying activity sequences, comparing sequences,
and formatting changes - used across notebooks and explanation classes.
"""

from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class SequenceChange:
    """Represents a single change between two sequences."""
    position: int  # 0-indexed position
    original: Optional[str]  # Original activity name (None if position didn't exist)
    changed: Optional[str]  # New activity name (None if position removed)
    change_type: str  # 'substitution', 'insertion', 'deletion'


class SequenceFormatter:
    """
    Formats activity sequences for display.

    Provides consistent sequence visualization across the project:
    - Single sequence display with positions
    - Side-by-side comparison of two sequences
    - Change highlighting and explanation

    Example:
        >>> formatter = SequenceFormatter(activity_names)
        >>> print(formatter.format_sequence([1, 5, 3]))
        [1] Assign seriousness → [2] Take in charge ticket → [3] Resolve ticket

        >>> print(formatter.format_comparison([1, 5, 3], [1, 5, 15]))
        Original:     [1] Assign seriousness → [2] Take in charge ticket → [3] Resolve ticket
        Counterfactual: [1] Assign seriousness → [2] Take in charge ticket → [3] Wait

        Changes:
          Position 3: Resolve ticket → Wait
    """

    def __init__(self, activity_names: List[str], padding_idx: int = 0):
        """
        Initialize formatter with activity vocabulary.

        Args:
            activity_names: List where activity_names[idx] = activity name
            padding_idx: Index used for padding (default 0)
        """
        self.activity_names = activity_names
        self.padding_idx = padding_idx

    @classmethod
    def from_vocab_dict(cls, vocab_dict: Dict[str, int], padding_idx: int = 0) -> "SequenceFormatter":
        """
        Create formatter from a vocabulary dictionary.

        Args:
            vocab_dict: Dict mapping activity name -> index
            padding_idx: Index used for padding

        Returns:
            SequenceFormatter instance
        """
        max_idx = max(vocab_dict.values()) if vocab_dict else 0
        activity_names = ['<pad>'] * (max_idx + 1)
        for name, idx in vocab_dict.items():
            activity_names[idx] = name
        return cls(activity_names, padding_idx)

    def get_activity_name(self, idx: int) -> str:
        """Get activity name for an index."""
        if idx is None:
            return "<none>"
        if idx < 0 or idx >= len(self.activity_names):
            return f"<unknown:{idx}>"
        return self.activity_names[idx]

    def format_sequence(self,
                        sequence: List[int],
                        show_positions: bool = True,
                        separator: str = " → ",
                        include_padding: bool = False) -> str:
        """
        Format a single sequence for display.

        Args:
            sequence: List of activity indices
            show_positions: Include position numbers [1], [2], etc.
            separator: String between activities
            include_padding: Include padding tokens in output

        Returns:
            Formatted string representation
        """
        parts = []
        for i, idx in enumerate(sequence):
            if not include_padding and idx == self.padding_idx:
                continue
            name = self.get_activity_name(idx)
            if show_positions:
                parts.append(f"[{i+1}] {name}")
            else:
                parts.append(name)
        return separator.join(parts)

    def format_sequence_vertical(self,
                                  sequence: List[int],
                                  indent: str = "  ") -> str:
        """
        Format sequence vertically (one activity per line).

        Args:
            sequence: List of activity indices
            indent: Indentation for each line

        Returns:
            Multi-line string
        """
        lines = []
        for i, idx in enumerate(sequence):
            if idx == self.padding_idx:
                continue
            name = self.get_activity_name(idx)
            lines.append(f"{indent}[{i+1}] {name}")
        return "\n".join(lines)

    def compute_changes(self,
                        original: List[int],
                        changed: List[int]) -> List[SequenceChange]:
        """
        Compute list of changes between two sequences.

        Args:
            original: Original sequence
            changed: Changed sequence

        Returns:
            List of SequenceChange objects describing each change
        """
        changes = []
        max_len = max(len(original), len(changed))

        for i in range(max_len):
            orig_idx = original[i] if i < len(original) else None
            new_idx = changed[i] if i < len(changed) else None

            # Skip if same
            if orig_idx == new_idx:
                continue

            # Skip padding comparisons
            if orig_idx == self.padding_idx:
                orig_idx = None
            if new_idx == self.padding_idx:
                new_idx = None

            if orig_idx == new_idx:
                continue

            # Determine change type
            if orig_idx is None:
                change_type = 'insertion'
            elif new_idx is None:
                change_type = 'deletion'
            else:
                change_type = 'substitution'

            changes.append(SequenceChange(
                position=i,
                original=self.get_activity_name(orig_idx) if orig_idx is not None else None,
                changed=self.get_activity_name(new_idx) if new_idx is not None else None,
                change_type=change_type
            ))

        return changes

    def format_changes(self,
                       changes: List[SequenceChange],
                       indent: str = "  ") -> str:
        """
        Format a list of changes for display.

        Args:
            changes: List of SequenceChange objects
            indent: Indentation for each line

        Returns:
            Formatted string describing changes
        """
        if not changes:
            return f"{indent}No changes"

        lines = []
        for change in changes:
            pos = change.position + 1  # 1-indexed for display

            if change.change_type == 'substitution':
                lines.append(f"{indent}Position {pos}: {change.original} → {change.changed}")
            elif change.change_type == 'insertion':
                lines.append(f"{indent}Position {pos}: <inserted> {change.changed}")
            elif change.change_type == 'deletion':
                lines.append(f"{indent}Position {pos}: {change.original} <removed>")

        return "\n".join(lines)

    def format_comparison(self,
                          original: List[int],
                          changed: List[int],
                          original_label: str = "Original",
                          changed_label: str = "Changed",
                          show_changes: bool = True) -> str:
        """
        Format a side-by-side comparison of two sequences.

        Args:
            original: Original sequence
            changed: Changed sequence
            original_label: Label for original sequence
            changed_label: Label for changed sequence
            show_changes: Include change summary

        Returns:
            Formatted comparison string
        """
        lines = []

        # Format both sequences
        orig_str = self.format_sequence(original, show_positions=True)
        changed_str = self.format_sequence(changed, show_positions=True)

        # Align labels
        max_label = max(len(original_label), len(changed_label))
        lines.append(f"{original_label:>{max_label}}: {orig_str}")
        lines.append(f"{changed_label:>{max_label}}: {changed_str}")

        if show_changes:
            changes = self.compute_changes(original, changed)
            if changes:
                lines.append("")
                lines.append("Changes:")
                lines.append(self.format_changes(changes))

        return "\n".join(lines)

    def format_counterfactual_explanation(self,
                                          original: List[int],
                                          counterfactual: List[int],
                                          original_prediction: str,
                                          counterfactual_prediction: str,
                                          counterfactual_probability: float,
                                          target: str) -> str:
        """
        Format a complete counterfactual explanation.

        Args:
            original: Original sequence
            counterfactual: Counterfactual sequence
            original_prediction: Prediction on original
            counterfactual_prediction: Prediction on counterfactual
            counterfactual_probability: Probability of counterfactual prediction
            target: Target activity name

        Returns:
            Formatted explanation string
        """
        lines = [
            "=" * 60,
            "COUNTERFACTUAL EXPLANATION",
            "=" * 60,
            "",
            "ORIGINAL:",
            f"  Sequence: {self.format_sequence(original, show_positions=False)}",
            f"  Length: {len([x for x in original if x != self.padding_idx])}",
            f"  Prediction: {original_prediction}",
            "",
            "COUNTERFACTUAL:",
            f"  Sequence: {self.format_sequence(counterfactual, show_positions=False)}",
            f"  Length: {len([x for x in counterfactual if x != self.padding_idx])}",
            f"  Prediction: {counterfactual_prediction} ({counterfactual_probability:.1%})",
            "",
            f"TARGET: {target}",
            "",
            "CHANGES:",
        ]

        changes = self.compute_changes(original, counterfactual)
        lines.append(self.format_changes(changes, indent="  "))

        return "\n".join(lines)


# Convenience functions for quick usage without instantiating formatter

def format_sequence(sequence: List[int],
                    activity_names: List[str],
                    show_positions: bool = True,
                    separator: str = " → ") -> str:
    """
    Quick function to format a sequence.

    Args:
        sequence: List of activity indices
        activity_names: List mapping index -> name
        show_positions: Include position numbers
        separator: String between activities

    Returns:
        Formatted string
    """
    formatter = SequenceFormatter(activity_names)
    return formatter.format_sequence(sequence, show_positions, separator)


def format_comparison(original: List[int],
                      changed: List[int],
                      activity_names: List[str],
                      original_label: str = "Original",
                      changed_label: str = "Changed") -> str:
    """
    Quick function to format a sequence comparison.

    Args:
        original: Original sequence
        changed: Changed sequence
        activity_names: List mapping index -> name
        original_label: Label for original
        changed_label: Label for changed

    Returns:
        Formatted comparison string
    """
    formatter = SequenceFormatter(activity_names)
    return formatter.format_comparison(original, changed, original_label, changed_label)


def print_sequence(sequence: List[int],
                   activity_names: List[str],
                   label: str = "Sequence",
                   show_positions: bool = True) -> None:
    """
    Print a sequence with a label.

    Args:
        sequence: List of activity indices
        activity_names: List mapping index -> name
        label: Label to print before sequence
        show_positions: Include position numbers
    """
    formatter = SequenceFormatter(activity_names)
    print(f"{label}: {formatter.format_sequence(sequence, show_positions)}")


def print_comparison(original: List[int],
                     changed: List[int],
                     activity_names: List[str],
                     original_label: str = "Original",
                     changed_label: str = "Changed") -> None:
    """
    Print a sequence comparison.

    Args:
        original: Original sequence
        changed: Changed sequence
        activity_names: List mapping index -> name
        original_label: Label for original
        changed_label: Label for changed
    """
    formatter = SequenceFormatter(activity_names)
    print(formatter.format_comparison(original, changed, original_label, changed_label))
