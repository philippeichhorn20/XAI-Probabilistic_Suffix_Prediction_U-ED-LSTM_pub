"""Configuration for BPIC17 dataset interpretability analysis."""

from .base_config import BaseConfig


CONFIG = BaseConfig(
    # Dataset identification
    dataset_name="BPIC17",

    # Feature configuration
    concept_name="concept:name",
    all_cat=["concept:name", "Action", "org:resource", "EventOrigin", "lifecycle:transition",
             "case:LoanGoal", "case:ApplicationType", "Accepted", "Selected"],
    all_num=["case_elapsed_time", "event_elapsed_time", "day_in_week", "seconds_in_day",
             "case:RequestedAmount", "FirstWithdrawalAmount", "NumberOfTerms", "MonthlyCost", "CreditScore"],
    growing_num_values=["case_elapsed_time"],

    # Paths (relative to project root)
    model_path="src/notebooks/training_variational_dropout/BPIC17/BPIC_2017_full_grad_norm_new_4layer.pkl",
    test_data_path="encoded_data/BPIC_2017_all_5_test.pkl",
    results_dir="src/interpretability/notebooks/results",

    # Analysis parameters
    ig_steps=50,
    seed=42,
)
