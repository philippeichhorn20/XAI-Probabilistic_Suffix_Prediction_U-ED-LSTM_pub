"""Configuration for Sepsis dataset interpretability analysis."""

from .base_config import BaseConfig


# No improved-pipeline model has been trained for Sepsis yet — improved fields
# are None, so flipping CONFIG.use_improved = True will raise a clear error
# pointing at the missing config until those pkls exist.
CONFIG = BaseConfig(
    # Dataset identification
    dataset_name="Sepsis",

    # Feature configuration
    concept_name="concept:name",
    all_cat=["concept:name", "org:group", "lifecycle:transition"],
    all_num=["case_elapsed_time", "event_elapsed_time"],
    growing_num_values=["case_elapsed_time"],



    # Paths (relative to project root)
    #model_path_old="src/notebooks/training_variational_dropout/Sepsis/Sepsis_full_grad_norm_2layer.pkl",

    #model_path_old="src/interpretability/improved_pipeline/camargo/sepsis/old/Training/pkl/Sepsis_full_grad_norm_new_4layer_from_git.pkl",
    model_path_old="src/interpretability/improved_pipeline/henryk/sepsis/old/Training/Sepsis_full_grad_norm_new_4layer.pkl",

    # Class-weighted CE + lower teacher forcing retrain (see improved/Training notebook).
    model_path_improved= "src/interpretability/improved_pipeline/henryk/sepsis/improved/Training/Sepsis_full_grad_norm_new_4layer_weighted_ce.pkl", #"src/interpretability/improved_pipeline/henryk/sepsis/improved/Training/Sepsis_full_grad_norm_new_4layer_weighted_ce.pkl",


    test_data_path_old="encoded_data/Sepsis_all_5_test.pkl",
    # Improved training reuses the same encoded test pkl (just a different model fit).
    test_data_path_improved="src/interpretability/improved_pipeline/henryk/sepsis/old/Loader/Sepsis_all_5_test.pkl",

    results_dir="src/interpretability/notebooks/results",

    # Variant switch — set CONFIG.use_improved = True in a notebook to load the improved variant.
    use_improved=False,

    # Analysis parameters
    ig_steps=50,
    seed=42,

    # --- Camargo baseline (under src/interpretability/improved_pipeline/camargo/) ---
    # The `_improved` siblings point at the all-features SharedCat retrain (see
    # improved_pipeline/camargo/sepsis/improved/). Notebooks toggle which one
    # is loaded via CONFIG.use_improved.
    camargo_model_pickle=(
        "src/interpretability/improved_pipeline/camargo/sepsis/old/Training/pkl/Sepsis_camargo_sharedcat_ngram5.pkl"
    ),
    camargo_test_pickle=(
        "src/interpretability/improved_pipeline/camargo/sepsis/old/Loader/pkl/Sepsis_all_5_roles_test.pkl"
    ),
    camargo_cat_indices=(0, 1),        # concept:name, org:group (old loader keeps only 2 cats)
    camargo_num_indices=(0,),          # case_elapsed_time (log-normalised by the loader)
    camargo_activity_feature="concept:name",
    camargo_model_class="SharedCat_LSTM",
    camargo_ngram_size=5,
    camargo_display_name="Sepsis",

    # Improved variant: ALL Sepsis features. The improved loader writes 26
    # categoricals (concept:name, org:group, lifecycle:transition, +23 clinical
    # flags incl. Diagnose) and 8 numericals; case_elapsed_time is appended
    # last in `continuous_columns + continuous_positive_columns`, so it sits at
    # numerical index 7. We pass through the entire feature set (no projection).
    camargo_model_pickle_improved=(
        "src/interpretability/improved_pipeline/camargo/sepsis/improved/Training/pkl/"
        "Sepsis_camargo_sharedcat_allfeat_ngram5.pkl"
    ),
    camargo_test_pickle_improved=(
        "src/interpretability/improved_pipeline/camargo/sepsis/improved/Loader/pkl/"
        "Sepsis_all_5_allfeat_test.pkl"
    ),
    camargo_cat_indices_improved=tuple(range(26)),  # all 26 categoricals
    camargo_num_indices_improved=tuple(range(8)),   # all 8 numericals
    camargo_model_class_improved="SharedCat_LSTM",
    camargo_ngram_size_improved=5,
)
