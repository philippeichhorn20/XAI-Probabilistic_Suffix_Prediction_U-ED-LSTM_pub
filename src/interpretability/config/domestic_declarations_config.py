"""Configuration for DomesticDeclarations dataset interpretability analysis."""

from .base_config import BaseConfig


CONFIG = BaseConfig(
    # Dataset identification
    dataset_name="DomesticDeclarations",

    # Feature configuration
    concept_name="Activity",
    all_cat=["Activity", "Resource"],
    all_num=["case_elapsed_time", "event_elapsed_time"],
    growing_num_values=["case_elapsed_time"],

    # Paths (relative to project root)
    model_path="src/notebooks/training_variational_dropout/DomesticDeclarations/DomesticDeclarations_full_grad_norm_4layer.pkl",
    test_data_path="encoded_data/test_philipp/domestic_declarations_all_5_test.pkl",
    results_dir="src/interpretability/notebooks/results",

    # Analysis parameters
    ig_steps=50,
    seed=42,
)
