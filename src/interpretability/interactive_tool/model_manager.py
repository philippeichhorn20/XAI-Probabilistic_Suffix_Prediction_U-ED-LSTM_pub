"""Model and data loading utilities."""

import torch
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass

from .config import ModelConfig, get_model_config


@dataclass
class CategoryInfo:
    """Information about a categorical feature."""
    name: str
    num_values: int
    value_to_idx: Dict[str, int]
    idx_to_value: Dict[int, str]


@dataclass
class NumericalInfo:
    """Information about a numerical feature."""
    name: str


class ModelManager:
    """Manages model loading and provides access to model metadata."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._loaded_models: Dict[str, Any] = {}
        self._loaded_datasets: Dict[str, Any] = {}
        self._loaded_eval_helpers: Dict[str, Any] = {}

    def load_model(self, model_name: str):
        """Load a model by name."""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        config = get_model_config(model_name)
        model_path = config.get_model_path(self.project_root)

        # Import here to avoid circular imports
        from src.model.dropout_uncertainty_enc_dec_LSTM.dropout_uncertainty_model import (
            DropoutUncertaintyEncoderDecoderLSTM
        )

        model = DropoutUncertaintyEncoderDecoderLSTM.load(str(model_path), dropout=0.0)
        model.eval()

        self._loaded_models[model_name] = model
        return model

    def load_test_dataset(self, model_name: str):
        """Load test dataset for a model."""
        if model_name in self._loaded_datasets:
            return self._loaded_datasets[model_name]

        config = get_model_config(model_name)
        data_path = config.get_test_data_path(self.project_root)

        dataset = torch.load(str(data_path), weights_only=False)
        self._loaded_datasets[model_name] = dataset
        return dataset

    def get_eval_helper(self, model_name: str):
        """Get evaluation helper for a model."""
        if model_name in self._loaded_eval_helpers:
            return self._loaded_eval_helpers[model_name]

        from src.evaluation.evaluation import Evaluation

        model = self.load_model(model_name)
        dataset = self.load_test_dataset(model_name)
        config = get_model_config(model_name)

        eval_helper = Evaluation(
            model=model,
            dataset=dataset,
            concept_name=config.concept_name,
            growing_num_values=config.growing_num_values,
            all_cat=config.all_cat,
            all_num=config.all_num
        )

        self._loaded_eval_helpers[model_name] = eval_helper
        return eval_helper

    def get_case_names(self, model_name: str) -> List[str]:
        """Get list of case names for a model's test dataset."""
        eval_helper = self.get_eval_helper(model_name)
        return sorted(eval_helper.cases.keys())

    def get_case_data(self, model_name: str, case_name: str) -> Tuple[Any, Any]:
        """Get case data (categorical, numerical tensors)."""
        eval_helper = self.get_eval_helper(model_name)
        if case_name not in eval_helper.cases:
            raise ValueError(f"Case '{case_name}' not found")
        # Return only categorical and numerical tensors (first two elements)
        # The event tuple contains (cat_tensors, num_tensors, case_name, ...)
        event = eval_helper.cases[case_name]
        return event[0], event[1]

    def get_categorical_info(self, model_name: str) -> List[CategoryInfo]:
        """Get information about categorical features."""
        model = self.load_model(model_name)
        cat_features = model.data_set_categories[0]

        result = []
        for feat_name, num_values, value_mapping in cat_features:
            idx_to_value = {v: k for k, v in value_mapping.items()}
            result.append(CategoryInfo(
                name=feat_name,
                num_values=num_values,
                value_to_idx=value_mapping,
                idx_to_value=idx_to_value
            ))
        return result

    def get_numerical_info(self, model_name: str) -> List[NumericalInfo]:
        """Get information about numerical features."""
        model = self.load_model(model_name)
        num_features = model.data_set_categories[1]

        result = []
        for feat_name, _, _ in num_features:
            result.append(NumericalInfo(name=feat_name))
        return result

    def get_encoder_features(self, model_name: str) -> Tuple[List[str], List[str]]:
        """Get encoder feature names (categorical, numerical)."""
        model = self.load_model(model_name)
        return list(model.enc_feat[0]), list(model.enc_feat[1])

    def get_decoder_features(self, model_name: str) -> Tuple[List[str], List[str]]:
        """Get decoder feature names (categorical, numerical)."""
        model = self.load_model(model_name)
        return list(model.dec_feat[0]), list(model.dec_feat[1])

    def get_activity_mapping(self, model_name: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Get activity name to index mapping and reverse."""
        config = get_model_config(model_name)
        cat_info = self.get_categorical_info(model_name)

        for info in cat_info:
            if info.name == config.concept_name:
                return info.value_to_idx, info.idx_to_value

        raise ValueError(f"Activity feature '{config.concept_name}' not found")

    def get_window_size(self, model_name: str) -> int:
        """Get the window size (padded sequence length) for the model's dataset."""
        dataset = self.load_test_dataset(model_name)
        return dataset.encoder_decoder.window_size

    def get_case_length(self, model_name: str, case_name: str) -> int:
        """Get the length of a case (number of events)."""
        case_data = self.get_case_data(model_name, case_name)
        cat_data = case_data[0]
        # Use first categorical feature to determine length
        return len(cat_data[0])
