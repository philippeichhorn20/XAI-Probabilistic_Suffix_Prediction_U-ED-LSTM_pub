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
    model_path_old="src/interpretability/improved_pipeline/henryk/bpic17/old/Training/BPIC_2017_remote_run_20260425_145308_epoch100.pkl",
    model_path_improved="src/interpretability/improved_pipeline/henryk/bpic17/improved/BPIC_2017_full_grad_norm_M_model_epoch81.pkl",

    test_data_path_old="encoded_data/BPIC_2017_all_5_test.pkl",
    test_data_path_improved="src/interpretability/improved_pipeline/henryk/bpic17/improved/Loader/pkl/BPIC_2017_all_5_test.pkl",

    results_dir="src/interpretability/notebooks/results",

    # Variant switch — set CONFIG.use_improved = True in a notebook to load the improved variant.
    use_improved=False,

    # Analysis parameters
    ig_steps=50,
    seed=42,

    # --- Camargo baseline (under src/interpretability/improved_pipeline/camargo/) ---
    camargo_model_pickle=(
        "src/interpretability/improved_pipeline/camargo/bpic17/old/Training/pkl/"
        "BPIC17_camargo_ngram15.pkl"
    ),
    camargo_test_pickle=(
        "src/interpretability/improved_pipeline/camargo/bpic17/old/Loader/pkl/"
        "BPIC_2017_all_5_test.pkl"
    ),
    camargo_cat_indices=(0, 2),        # concept:name, org:resource
    camargo_num_indices=(0,),          # case_elapsed_time
    camargo_activity_feature="concept:name",
    camargo_model_class="FullShared_Join_LSTM",
    camargo_ngram_size=15,
    camargo_display_name="BPIC17 (n-gram=15)",
)
