"""Configuration for Helpdesk dataset interpretability analysis."""

from .base_config import BaseConfig


CONFIG = BaseConfig(
    # Dataset identification
    dataset_name="Helpdesk",

    # Feature configuration
    concept_name="Activity",
    all_cat=["Activity", "Resource"],
    all_num=["case_elapsed_time", "event_elapsed_time"],
    growing_num_values=["case_elapsed_time"],

    # Paths (relative to project root)

    #model_path = "src/interpretability/improved_pipeline/henryk/helpdesk/improved/Training/pkl/Helpdesk_full_grad_norm_improved_henryk-.pkl", # improved

    model_path="src/interpretability/improved_pipeline/henryk/helpdesk/old/Training/Helpdesk_full_grad_norm_philipp_4layer_philipp_final_run.pkl", #legacy
    # NOTE: must match the vocabulary the model was trained on. Improved-loader train/test pkls
    # have only 12 Activity classes (rare activities like DUPLICATE/INVALID/RESOLVED/VERIFIED
    # didn't appear in the improved-loader train split, so the encoder doesn't know them);
    # legacy test_philipp pkl has 16 -> embedding-index-out-of-range error.
    #test_data_path="src/interpretability/improved_pipeline/henryk/helpdesk/improved/Loader/pkl/helpdesk_all_5_test.pkl", # improved
    test_data_path="encoded_data/test_philipp/helpdesk_all_5_test.pkl",  # legacy; only valid for the legacy model
    results_dir="src/interpretability/notebooks/results",

    # Analysis parameters
    ig_steps=50,
    seed=42,
)
