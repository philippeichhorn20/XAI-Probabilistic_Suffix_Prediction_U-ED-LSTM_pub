"""Configuration for Helpdesk dataset interpretability analysis."""

from .base_config import BaseConfig


# NOTE: model and test pkl must match on vocabulary. The improved-loader
# train/test pkls have only 12 Activity classes (rare activities like
# DUPLICATE/INVALID/RESOLVED/VERIFIED didn't appear in the improved-loader
# train split, so the encoder doesn't know them); the legacy test_philipp
# pkl has 16. Mixing variants -> embedding-index-out-of-range error.
CONFIG = BaseConfig(
    # Dataset identification
    dataset_name="Helpdesk",

    # Feature configuration
    concept_name="Activity",
    all_cat=["Activity", "Resource"],
    all_num=["case_elapsed_time", "event_elapsed_time"],
    growing_num_values=["case_elapsed_time"],

    # Paths (relative to project root)

    model_path_old="src/interpretability/improved_pipeline/henryk/helpdesk/old/Training/Helpdesk_full_grad_norm_philipp_4layer_philipp_final_run.pkl",
    model_path_improved="src/interpretability/improved_pipeline/henryk/helpdesk/improved/Training/pkl/Helpdesk_full_grad_norm_improved_henryk-p1oversampling.pkl", #"src/interpretability/improved_pipeline/henryk/helpdesk/improved/Training/pkl/-.pkl",

    test_data_path_old="encoded_data/test_philipp/helpdesk_all_5_test.pkl",
    test_data_path_improved="src/interpretability/improved_pipeline/henryk/helpdesk/improved/Loader/pkl/helpdesk_all_5_test.pkl",

    results_dir="src/interpretability/notebooks/results",

    # Variant switch — set CONFIG.use_improved = True in a notebook to load the improved variant.
    use_improved=False,

    # Analysis parameters
    ig_steps=50,
    seed=42,

    # --- Camargo baseline (under src/interpretability/improved_pipeline/camargo/) ---
    camargo_model_pickle=(
        "src/interpretability/improved_pipeline/camargo/helpdesk/old/Training/pkl/"
        "Helpdesk_camargo_sharedcat_roles_ngram5.pkl"
    ),
    camargo_test_pickle=(
        "src/interpretability/improved_pipeline/camargo/helpdesk/old/Loader/pkl/"
        "helpdesk_all_5_roles_test.pkl"
    ),
    camargo_cat_indices=(0, 1),        # Activity, Role
    camargo_num_indices=(0,),          # case_elapsed_time
    camargo_activity_feature="Activity",
    camargo_model_class="SharedCat_LSTM",
    camargo_ngram_size=5,
    camargo_display_name="Helpdesk",
)
