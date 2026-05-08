"""Configuration for DomesticDeclarations dataset interpretability analysis."""

from .base_config import BaseConfig


# No improved-pipeline model has been trained for DomesticDeclarations yet —
# improved fields are None, so flipping CONFIG.use_improved = True will raise
# a clear error pointing at the missing config until those pkls exist.
CONFIG = BaseConfig(
    # Dataset identification
    dataset_name="DomesticDeclarations",

    # Feature configuration
    concept_name="Activity",
    all_cat=["Activity", "Role"],
    all_num=["case_elapsed_time", "event_elapsed_time"],
    growing_num_values=["case_elapsed_time"],

    # Paths (relative to project root)
    model_path_old="src/interpretability/improved_pipeline/camargo/domestic_declarations/old/Training/pkl/DomesticDeclarations_camargo_sharedcat_role_ngram5.pkl",
    model_path_improved=None,

    test_data_path_old="encoded_data/test_philipp/domestic_declarations_all_5_test.pkl",
    test_data_path_improved=None,

    results_dir="src/interpretability/notebooks/results",

    # Variant switch — set CONFIG.use_improved = True in a notebook to load the improved variant.
    use_improved=False,

    # Analysis parameters
    ig_steps=50,
    seed=42,

    # --- Camargo baseline (under src/interpretability/improved_pipeline/camargo/) ---
    camargo_model_pickle=(
        "src/interpretability/improved_pipeline/camargo/domestic_declarations/old/Training/pkl/"
        "DomesticDeclarations_camargo_sharedcat_role_ngram5.pkl"
    ),
    camargo_test_pickle=(
        "src/interpretability/improved_pipeline/camargo/domestic_declarations/old/Loader/pkl/"
        "domestic_declarations_all_5_roles_test.pkl"
    ),
    camargo_cat_indices=(0, 1),        # Activity, Role
    camargo_num_indices=(0,),          # case_elapsed_time
    camargo_activity_feature="Activity",
    camargo_model_class="SharedCat_LSTM",
    camargo_ngram_size=5,
    camargo_display_name="DomesticDeclarations",
)
