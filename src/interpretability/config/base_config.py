"""Base configuration class for interpretability notebooks."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaseConfig:
    """
    Base configuration for interpretability analysis.

    Subclass this for dataset-specific configurations.
    """

    # Dataset identification
    dataset_name: str = "Unknown"

    # Feature configuration
    concept_name: str = "Activity"  # Main categorical feature to predict
    all_cat: List[str] = field(default_factory=lambda: ["Activity"])
    all_num: List[str] = field(default_factory=list)
    growing_num_values: List[str] = field(default_factory=list)

    # Paths (relative to project root)
    model_path: str = ""
    test_data_path: str = ""
    results_dir: str = "src/interpretability/notebooks/results"

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

    def get_model_path(self) -> Path:
        """Get absolute path to model file."""
        return self.project_root / self.model_path

    def get_test_data_path(self) -> Path:
        """Get absolute path to test data file."""
        return self.project_root / self.test_data_path

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
            'model_path': self.model_path,
            'test_data_path': self.test_data_path,
            'ig_steps': self.ig_steps,
            'seed': self.seed,
        }

    def validate(self) -> bool:
        """Validate that required paths exist."""
        model_path = self.get_model_path()
        data_path = self.get_test_data_path()

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Test data not found: {data_path}")

        return True
