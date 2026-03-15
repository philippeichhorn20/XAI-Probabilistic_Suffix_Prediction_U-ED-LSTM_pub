"""REVISED+: Counterfactual Explanations with MP-Declare constraints."""

from .log_utils import csv_to_xes
from .simple_vae import SimpleSequenceVAE, train_vae
from .revised_plus import (
    RevisedPlusConfig,
    RevisedPlusCounterfactual,
    RevisedPlusExplanation,
    RevisedPlus,
    RevisedPlusModelPredictor,
    create_revised_plus_for_model,
)

# Optional imports that require external dependencies
try:
    from .mp_declare import discover_constraints, check_conformance, load_event_log
    _MP_DECLARE_AVAILABLE = True
except ImportError:
    _MP_DECLARE_AVAILABLE = False

try:
    from .rum_mpdeclare import discover_mpdeclare, MPDeclareConstraint, mpdeclare_to_declare_constraints, RUM_TO_DECLARE_TEMPLATE
    _RUM_AVAILABLE = True
except ImportError:
    _RUM_AVAILABLE = False
