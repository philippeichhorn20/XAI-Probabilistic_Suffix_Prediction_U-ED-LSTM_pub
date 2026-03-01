"""Case editing utilities for manipulating process cases."""

import torch
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class Event:
    """Represents a single event in a process case."""
    categorical: Dict[str, int]  # feature_name -> encoded value
    numerical: Dict[str, float]  # feature_name -> value

    def copy(self) -> 'Event':
        return Event(
            categorical=self.categorical.copy(),
            numerical=self.numerical.copy()
        )


@dataclass
class EditableCase:
    """A case that can be edited."""
    events: List[Event]
    cat_feature_names: List[str]
    num_feature_names: List[str]
    original_length: int

    def copy(self) -> 'EditableCase':
        return EditableCase(
            events=[e.copy() for e in self.events],
            cat_feature_names=self.cat_feature_names.copy(),
            num_feature_names=self.num_feature_names.copy(),
            original_length=self.original_length
        )

    def __len__(self) -> int:
        return len(self.events)

    def add_event(self, event: Optional[Event] = None):
        """Add an event at the end."""
        if event is None:
            # Create empty event with default values
            event = Event(
                categorical={name: 0 for name in self.cat_feature_names},
                numerical={name: 0.0 for name in self.num_feature_names}
            )
        self.events.append(event)

    def remove_event(self, index: int):
        """Remove event at index."""
        if 0 <= index < len(self.events):
            self.events.pop(index)

    def get_prefix(self, length: int) -> 'EditableCase':
        """Get first 'length' events as a new EditableCase."""
        return EditableCase(
            events=[e.copy() for e in self.events[:length]],
            cat_feature_names=self.cat_feature_names.copy(),
            num_feature_names=self.num_feature_names.copy(),
            original_length=self.original_length
        )


class CaseEditor:
    """Handles conversion between tensor format and editable format."""

    def __init__(self, cat_feature_names: List[str], num_feature_names: List[str]):
        self.cat_feature_names = cat_feature_names
        self.num_feature_names = num_feature_names

    def tensor_to_editable(
        self,
        case_data: Tuple[List[torch.Tensor], List[torch.Tensor]],
        activity_feature: str = "Activity"
    ) -> EditableCase:
        """Convert tensor case data to editable format, filtering out padding."""
        cat_tensors, num_tensors = case_data

        # Determine case length from first categorical tensor
        case_length = len(cat_tensors[0])

        # Find the activity feature index to detect padding
        activity_idx = None
        for j, name in enumerate(self.cat_feature_names):
            if name == activity_feature:
                activity_idx = j
                break

        events = []
        for i in range(case_length):
            # Skip padding events (where activity index is 0)
            if activity_idx is not None and activity_idx < len(cat_tensors):
                if cat_tensors[activity_idx][i].item() == 0:
                    continue

            cat_values = {}
            for j, name in enumerate(self.cat_feature_names):
                if j < len(cat_tensors):
                    cat_values[name] = cat_tensors[j][i].item()
                else:
                    cat_values[name] = 0

            num_values = {}
            for j, name in enumerate(self.num_feature_names):
                if j < len(num_tensors):
                    num_values[name] = num_tensors[j][i].item()
                else:
                    num_values[name] = 0.0

            events.append(Event(categorical=cat_values, numerical=num_values))

        return EditableCase(
            events=events,
            cat_feature_names=self.cat_feature_names,
            num_feature_names=self.num_feature_names,
            original_length=len(events)
        )

    def editable_to_tensors(self, case: EditableCase) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Convert editable case back to tensor format."""
        cat_tensors = []
        for name in self.cat_feature_names:
            values = [e.categorical.get(name, 0) for e in case.events]
            cat_tensors.append(torch.tensor(values, dtype=torch.long))

        num_tensors = []
        for name in self.num_feature_names:
            values = [e.numerical.get(name, 0.0) for e in case.events]
            num_tensors.append(torch.tensor(values, dtype=torch.float32))

        return cat_tensors, num_tensors

    def create_prefix_suffix_split(
        self,
        case: EditableCase,
        prefix_length: int
    ) -> Tuple[Tuple[List[torch.Tensor], List[torch.Tensor]],
               Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Split case into prefix and suffix tensors."""
        prefix_case = case.get_prefix(prefix_length)
        suffix_case = EditableCase(
            events=case.events[prefix_length:],
            cat_feature_names=case.cat_feature_names,
            num_feature_names=case.num_feature_names,
            original_length=case.original_length
        )

        prefix_tensors = self.editable_to_tensors(prefix_case)
        suffix_tensors = self.editable_to_tensors(suffix_case)

        return prefix_tensors, suffix_tensors

    def get_activity_sequence(
        self,
        case: EditableCase,
        activity_feature: str,
        idx_to_activity: Dict[int, str]
    ) -> List[str]:
        """Get human-readable activity sequence from case."""
        activities = []
        for event in case.events:
            act_idx = event.categorical.get(activity_feature, 0)
            act_name = idx_to_activity.get(act_idx, f"Unknown_{act_idx}")
            activities.append(act_name)
        return activities


def detect_constant_features(case: EditableCase) -> Tuple[Set[str], Set[str]]:
    """Detect features that have the same value across all events in the case.

    Returns:
        (constant_cat, constant_num): Sets of feature names that are constant
    """
    if not case.events:
        return set(), set()

    constant_cat: Set[str] = set()
    constant_num: Set[str] = set()

    for feat_name in case.cat_feature_names:
        values = [e.categorical.get(feat_name) for e in case.events]
        if len(set(values)) == 1:
            constant_cat.add(feat_name)

    for feat_name in case.num_feature_names:
        values = [e.numerical.get(feat_name, 0.0) for e in case.events]
        if values and max(values) - min(values) < 1e-6:
            constant_num.add(feat_name)

    return constant_cat, constant_num


def get_case_level_features(model_name: str, case: EditableCase) -> Tuple[Set[str], Set[str]]:
    """Get case-level (constant) features, either from config or auto-detected.

    Returns:
        (case_level_cat, case_level_num): Sets of feature names
    """
    from .config import get_model_config

    config = get_model_config(model_name)

    if config.case_level_cat is not None:
        case_level_cat = set(config.case_level_cat)
    else:
        case_level_cat, _ = detect_constant_features(case)

    if config.case_level_num is not None:
        case_level_num = set(config.case_level_num)
    else:
        _, case_level_num = detect_constant_features(case)

    return case_level_cat, case_level_num
