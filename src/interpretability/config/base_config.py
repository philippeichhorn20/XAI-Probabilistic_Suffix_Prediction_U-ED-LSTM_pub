"""Base configuration class for interpretability notebooks."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BaseConfig:
    """
    Base configuration for interpretability analysis.

    Each dataset config exposes paired paths for the *old* (baseline) and
    *improved* models + their matching test pickles. Notebooks pick which
    one to load by toggling ``CONFIG.use_improved`` near the top:

        from src.interpretability.config.helpdesk_config import CONFIG
        CONFIG.use_improved = True   # or False
        # downstream calls to CONFIG.get_model_path() / get_test_data_path()
        # automatically resolve to the chosen variant.

    If the chosen variant is not configured for the dataset (e.g. no
    improved model has been trained yet), the getters raise ``ValueError``
    with a clear message instead of silently falling back.
    """

    # Dataset identification
    dataset_name: str = "Unknown"

    # Feature configuration
    concept_name: str = "Activity"  # Main categorical feature to predict
    all_cat: List[str] = field(default_factory=lambda: ["Activity"])
    all_num: List[str] = field(default_factory=list)
    growing_num_values: List[str] = field(default_factory=list)

    # Paths (relative to project root) — paired old / improved.
    # `None` means that variant has not been produced for this dataset yet.
    model_path_old: Optional[str] = None
    model_path_improved: Optional[str] = None
    test_data_path_old: Optional[str] = None
    test_data_path_improved: Optional[str] = None

    results_dir: str = "src/interpretability/notebooks/results"

    # Variant switch — flip to True in a notebook to load the improved model + test pkl.
    use_improved: bool = False

    # ------------------------------------------------------------------
    # Camargo baseline (used by src/interpretability/notebooks/camargo/*).
    # All paths point at the improved_pipeline tree so the camargo spec
    # lives in one place per dataset and the notebooks can fetch it from
    # CONFIG instead of hardcoding paths in `_camargo_utils.py`.
    #
    # The `_improved` siblings mirror the henryk-side pattern (cf.
    # `model_path_improved` / `test_data_path_improved`). When
    # `CONFIG.use_improved` is True, `spec_from_config` in
    # `_camargo_utils.py` picks the improved variant for any field that
    # is populated, falling back to the non-improved field otherwise.
    # ------------------------------------------------------------------
    camargo_model_pickle: Optional[str] = None
    camargo_test_pickle: Optional[str] = None
    camargo_cat_indices: Optional[Tuple[int, ...]] = None  # categorical positions consumed by the model
    camargo_num_indices: Optional[Tuple[int, ...]] = None  # numerical feature positions
    camargo_activity_feature: Optional[str] = None         # e.g. 'Activity' or 'concept:name'
    camargo_model_class: str = "FullShared_Join_LSTM"      # or 'SharedCat_LSTM'
    camargo_ngram_size: int = 0                             # >0 = paper n-gram window
    camargo_display_name: Optional[str] = None              # cosmetic name in DatasetSpec

    # Improved-variant overrides — set to None to fall back to the field above.
    camargo_model_pickle_improved: Optional[str] = None
    camargo_test_pickle_improved: Optional[str] = None
    camargo_cat_indices_improved: Optional[Tuple[int, ...]] = None
    camargo_num_indices_improved: Optional[Tuple[int, ...]] = None
    camargo_model_class_improved: Optional[str] = None
    camargo_ngram_size_improved: Optional[int] = None

    # Analysis parameters
    ig_steps: int = 50  # Integration steps for Integrated Gradients
    seed: int = 42      # Random seed for reproducibility

    # Project root (set automatically)
    _project_root: Optional[Path] = field(default=None, repr=False)

    def __post_init__(self):
        """Find project root after initialization."""
        if self._project_root is None:
            self._project_root = self._find_project_root()

    def _find_project_root(self) -> Path:
        """Find project root by looking for src directory."""
        current = Path(__file__).resolve()
        while current != current.parent:
            if (current / 'src').is_dir():
                return current
            current = current.parent
        # Fallback to current working directory
        return Path.cwd()

    @property
    def project_root(self) -> Path:
        """Get project root path."""
        if self._project_root is None:
            self._project_root = self._find_project_root()
        return self._project_root

    @property
    def variant(self) -> str:
        """Currently selected variant: 'improved' or 'old'."""
        return 'improved' if self.use_improved else 'old'

    def _resolve(self, attr_old: str, attr_improved: str, what: str) -> Path:
        chosen = getattr(self, attr_improved if self.use_improved else attr_old)
        if not chosen:
            raise ValueError(
                f"No {self.variant} {what} configured for dataset "
                f"'{self.dataset_name}'. Set "
                f"`{attr_improved if self.use_improved else attr_old}` in the "
                f"config, or flip CONFIG.use_improved."
            )
        return self.project_root / chosen

    def get_model_path(self) -> Path:
        """Return the absolute model-pkl path for the currently selected variant."""
        return self._resolve('model_path_old', 'model_path_improved', 'model')

    def get_test_data_path(self) -> Path:
        """Return the absolute test-pkl path for the currently selected variant."""
        return self._resolve('test_data_path_old', 'test_data_path_improved', 'test data')

    def get_camargo_paths(self) -> Tuple[Path, Path]:
        """Return absolute (model_pickle, test_pickle) for the Camargo baseline.

        Honors ``self.use_improved``: when True, prefers the
        ``camargo_*_improved`` field if set, otherwise falls back to the
        non-improved field. Raises ValueError with a descriptive message
        if no path is configured.
        """
        model_rel, _ = self._resolve_camargo_field('model_pickle', required=True)
        test_rel, _ = self._resolve_camargo_field('test_pickle', required=True)
        return (self.project_root / model_rel, self.project_root / test_rel)

    def _resolve_camargo_field(self, base: str, required: bool = False):
        """Pick the improved value when ``use_improved`` is set and populated.

        Returns ``(value, used_improved)`` so callers can tell which variant
        they got. ``base`` is the field stem, e.g. ``'model_pickle'`` or
        ``'cat_indices'``; the lookup is ``camargo_<base>`` /
        ``camargo_<base>_improved``.
        """
        old = getattr(self, f'camargo_{base}')
        improved = getattr(self, f'camargo_{base}_improved', None)
        if self.use_improved and improved is not None:
            return improved, True
        if required and not old:
            raise ValueError(
                f"Camargo {base} not configured for dataset "
                f"'{self.dataset_name}'. Set camargo_{base}"
                + (f" / camargo_{base}_improved" if self.use_improved else "")
                + f" in {self.dataset_name.lower()}_config.py."
            )
        return old, False

    def get_results_dir(self) -> Path:
        """Get absolute path to results directory, creating if needed."""
        results_path = self.project_root / self.results_dir / self.dataset_name.lower()
        results_path.mkdir(parents=True, exist_ok=True)
        return results_path

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            'dataset_name': self.dataset_name,
            'concept_name': self.concept_name,
            'all_cat': self.all_cat,
            'all_num': self.all_num,
            'growing_num_values': self.growing_num_values,
            'model_path_old': self.model_path_old,
            'model_path_improved': self.model_path_improved,
            'test_data_path_old': self.test_data_path_old,
            'test_data_path_improved': self.test_data_path_improved,
            'use_improved': self.use_improved,
            'ig_steps': self.ig_steps,
            'seed': self.seed,
        }

    def validate(self) -> bool:
        """Validate that the currently selected variant's paths exist."""
        model_path = self.get_model_path()
        data_path = self.get_test_data_path()

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Test data not found: {data_path}")

        return True
