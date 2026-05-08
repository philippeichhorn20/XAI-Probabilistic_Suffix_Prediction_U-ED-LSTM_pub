"""Generate `old_vs_improved.ipynb` for every (author, dataset) pair under
`improved_pipeline/`. Run once after editing per-pair config; commit the
generated notebooks alongside.

    pipenv run python src/interpretability/improved_pipeline/_shared/generate_comparison_notebooks.py

Each notebook imports `comparison_helpers` from this directory, so the heavy
lifting (sampling, batch_evaluate, plotting) lives in exactly one module.

For pairs missing one side (e.g. camargo has no `improved/`, henryk/sepsis's
`improved/` has no Loader subdir), the parameter cell sets the missing path
to `None` and a TODO marker in markdown. The notebook still runs end-to-end
on the available side and skips the comparison cleanly.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]   # -> repo root
PIPELINE_ROOT = REPO_ROOT / "src" / "interpretability" / "improved_pipeline"


# --- Per-dataset event-label lists (used by EventLabelCountInterval) ------
HELPDESK_EVENT_LABELS = [
    "Assign seriousness", "Take in charge ticket", "Resolve ticket", "Closed",
    "Insert ticket", "Wait", "Create SW anomaly", "Require upgrade", "VERIFIED",
    "DUPLICATE", "Resolve SW anomaly", "Schedule intervention", "RESOLVED",
    "INVALID",
]

BPIC17_EVENT_LABELS = [
    "A_Accepted", "A_Cancelled", "A_Complete", "A_Concept", "A_Create Application",
    "A_Denied", "A_Incomplete", "A_Pending", "A_Submitted", "A_Validating",
    "O_Accepted", "O_Cancelled", "O_Create Offer", "O_Created", "O_Refused",
    "O_Returned", "O_Sent (mail and online)", "O_Sent (online only)",
    "W_Assess potential fraud", "W_Call after offers", "W_Call incomplete files",
    "W_Complete application", "W_Handle leads", "W_Personal Loan collection",
    "W_Shortened completion ", "W_Validate application",
]

SEPSIS_EVENT_LABELS = [
    "ER Registration", "Leucocytes", "CRP", "LacticAcid", "ER Triage",
    "ER Sepsis Triage", "IV Liquid", "IV Antibiotics", "Admission NC",
    "Release A", "Return ER", "Admission IC", "Release B", "Release C",
    "Release D", "Release E",
]

DOMESTIC_DECLARATIONS_EVENT_LABELS = [
    # TODO: fill in once the dataset's activity dict is known.
]


@dataclass
class PairConfig:
    author: str
    dataset: str
    caption: str
    concept_name: str
    activity_key: str            # passed to EventLabelCountInterval / DLS
    event_label_list: list[str]
    growing_num_values: list[str] = field(default_factory=lambda: ["case_elapsed_time"])
    all_cat: list[str] | None = None
    all_num: list[str] | None = None
    samples_per_case: int = 100
    num_processes: int = 16
    save_every: int = 100
    random_order: bool = False
    value_factor_time: float = 3600 * 24
    # Paths (None = doesn't exist yet)
    model_old_rel: str | None = None
    test_pkl_old_rel: str | None = None
    model_improved_rel: str | None = None
    test_pkl_improved_rel: str | None = None
    sampled_old_rel: str = "evaluation_results/old"
    sampled_improved_rel: str = "evaluation_results/improved"
    notes: str = ""


# Path conventions: all `*_rel` are relative to the pair directory
# (e.g. `improved_pipeline/henryk/helpdesk/`). The notebook resolves them at
# runtime from its own __file__ so it works regardless of where the kernel
# launched.

PAIRS: list[PairConfig] = [
    # ---- HENRYK pairs (improved exists) ----
    PairConfig(
        author="henryk", dataset="bpic17", caption="BPIC17",
        concept_name="concept:name",
        activity_key="concept:name",
        event_label_list=BPIC17_EVENT_LABELS,
        all_cat=["concept:name", "org:resource", "lifecycle:transition"],
        all_num=["case_elapsed_time", "event_elapsed_time"],
        samples_per_case=1000, num_processes=32, save_every=500,
        model_old_rel="old/Training/BPIC_2017_remote_run_20260425_145308_epoch100.pkl",
        # old/ has no test pkl — fall back to improved/Loader/pkl/ which is the
        # encoded-data-of-record. Override in the parameter cell if the old model
        # was trained against a different test split.
        test_pkl_old_rel="improved/Loader/pkl/BPIC_2017_all_5_test.pkl",
        model_improved_rel="improved/Training/pkl/BPIC_2017_full_grad_norm_philipp_final_run.pkl",
        test_pkl_improved_rel="improved/Loader/pkl/BPIC_2017_all_5_test.pkl",
        notes=("Improved model is the augment+sample variant being trained as of 2026-05. "
               "If improved checkpoint is not yet final, leave `MODEL_IMPROVED_PATH = None` "
               "and only the OLD side will be sampled/evaluated."),
    ),
    PairConfig(
        author="henryk", dataset="helpdesk", caption="Helpdesk",
        concept_name="Activity",
        activity_key="Activity",
        event_label_list=HELPDESK_EVENT_LABELS,
        all_cat=["Activity", "Resource"],
        all_num=["case_elapsed_time", "event_elapsed_time"],
        samples_per_case=100, num_processes=16, save_every=50,
        random_order=True,
        model_old_rel="old/Training/Helpdesk_full_grad_norm_philipp_4layer_philipp_final_run.pkl",
        test_pkl_old_rel="old/Loader/helpdesk_all_5_test.pkl",
        model_improved_rel="improved/Training/pkl/Helpdesk_full_grad_norm_improved_henryk.pkl",
        test_pkl_improved_rel="improved/Loader/pkl/helpdesk_all_5_test.pkl",
    ),
    PairConfig(
        author="henryk", dataset="sepsis", caption="Sepsis",
        concept_name="concept:name",
        activity_key="concept:name",
        event_label_list=SEPSIS_EVENT_LABELS,
        samples_per_case=100, num_processes=16, save_every=50,
        model_old_rel="old/Training/Sepsis_full_grad_norm_new_4layer.pkl",
        test_pkl_old_rel="old/Loader/Sepsis_all_5_test.pkl",
        model_improved_rel="improved/Training/Sepsis_full_grad_norm_new_4layer_weighted_tf03.pkl",
        # improved/ has no Loader subdir; reuse old/Loader test pkl since it's
        # the same dataset split.
        test_pkl_improved_rel="old/Loader/Sepsis_all_5_test.pkl",
        notes="Improved/ has no Loader/ subdir — both sides reuse old/Loader/Sepsis_all_5_test.pkl.",
    ),
    PairConfig(
        author="henryk", dataset="domestic_declarations", caption="Domestic Declarations",
        concept_name="concept:name",
        activity_key="concept:name",
        event_label_list=DOMESTIC_DECLARATIONS_EVENT_LABELS,
        samples_per_case=100, num_processes=16, save_every=50,
        model_old_rel="old/Training/DomesticDeclarations_full_grad_norm_4layer_philipp_unchanged.pkl",
        test_pkl_old_rel="old/Loader/domestic_declarations_all_5_test.pkl",
        model_improved_rel=None,
        test_pkl_improved_rel=None,
        notes=("No improved/ directory yet — only the OLD side will run. "
               "Fill EVENT_LABEL_LIST in the parameter cell once finalised."),
    ),
    # ---- CAMARGO pairs (no improved/ for any dataset) ----
    PairConfig(
        author="camargo", dataset="bpic17", caption="BPIC17 (Camargo)",
        concept_name="concept:name",
        activity_key="concept:name",
        event_label_list=BPIC17_EVENT_LABELS,
        all_cat=["concept:name", "org:resource", "lifecycle:transition"],
        all_num=["case_elapsed_time", "event_elapsed_time"],
        samples_per_case=1000, num_processes=32, save_every=500,
        model_old_rel="old/Training/pkl/BPIC17_camargo_ngram15.pkl",
        test_pkl_old_rel="old/Loader/pkl/BPIC_2017_all_5_test.pkl",
        model_improved_rel=None,
        test_pkl_improved_rel=None,
        notes=("Camargo uses a different model class than DropoutUncertaintyEncoderDecoderLSTM. "
               "Replace `ensure_sampled` with a Camargo-specific loader before running."),
    ),
    PairConfig(
        author="camargo", dataset="helpdesk", caption="Helpdesk (Camargo)",
        concept_name="Activity",
        activity_key="Activity",
        event_label_list=HELPDESK_EVENT_LABELS,
        samples_per_case=100, num_processes=16, save_every=50,
        random_order=True,
        model_old_rel="old/Training/pkl/Helpdesk_camargo_sharedcat_roles_ngram5.pkl",
        test_pkl_old_rel="old/Loader/pkl/helpdesk_all_5_roles_test.pkl",
        model_improved_rel=None,
        test_pkl_improved_rel=None,
        notes=("Camargo uses a different model class than DropoutUncertaintyEncoderDecoderLSTM. "
               "Replace `ensure_sampled` with a Camargo-specific loader before running."),
    ),
    PairConfig(
        author="camargo", dataset="sepsis", caption="Sepsis (Camargo)",
        concept_name="concept:name",
        activity_key="concept:name",
        event_label_list=SEPSIS_EVENT_LABELS,
        samples_per_case=100, num_processes=16, save_every=50,
        model_old_rel="old/Training/pkl/Sepsis_full_grad_norm_new_4layer_from_git.pkl",
        test_pkl_old_rel="old/Loader/pkl/Sepsis_all_5_test.pkl",
        model_improved_rel=None,
        test_pkl_improved_rel=None,
        notes="Camargo uses a different model class — replace ensure_sampled accordingly.",
    ),
    PairConfig(
        author="camargo", dataset="domestic_declarations", caption="Domestic Declarations (Camargo)",
        concept_name="concept:name",
        activity_key="concept:name",
        event_label_list=DOMESTIC_DECLARATIONS_EVENT_LABELS,
        samples_per_case=100, num_processes=16, save_every=50,
        # The Sepsis-named pkl in this dir is presumably misplaced — leave as TODO.
        model_old_rel=None,
        test_pkl_old_rel="old/Loader/pkl/domestic_declarations_all_5_test.pkl",
        model_improved_rel=None,
        test_pkl_improved_rel=None,
        notes=("Camargo + domestic_declarations: model checkpoint not yet placed correctly "
               "(the only pkl in old/Training/pkl/ is misnamed Sepsis_camargo_*). Fill in once known. "
               "EVENT_LABEL_LIST also TODO."),
    ),
]


# ---------------------------------------------------------------------------
# Notebook builder
# ---------------------------------------------------------------------------

def _md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _intro_md(cfg: PairConfig) -> str:
    notes = f"\n\n> **Notes:** {cfg.notes}" if cfg.notes else ""
    return (
        f"# {cfg.caption} — Old vs Improved\n"
        f"\n"
        f"Single-source comparison: runs MC suffix sampling for both checkpoints (skipping any side that\n"
        f"already has chunked outputs), evaluates the same metric set Henryk uses in\n"
        f"`evaluation_metric_notebooks/improved/`, then plots them side by side.\n"
        f"\n"
        f"**Pair:** `{cfg.author}/{cfg.dataset}` &nbsp;·&nbsp; **Activity key:** `{cfg.activity_key}` &nbsp;·&nbsp; "
        f"**Samples/case:** `{cfg.samples_per_case}`{notes}\n"
        f"\n"
        f"## How to run\n"
        f"\n"
        f"1. Confirm the paths in the parameter cell point at the right checkpoints + test pickles.\n"
        f"2. Execute top-to-bottom. Sampling is skipped automatically if `SAMPLED_DIR_*` already contains `results_part_*.pkl`.\n"
        f"3. The bottom cells (table + overlay plot) tolerate either side being `None` — useful while the improved\n"
        f"   checkpoint is still being trained.\n"
    )


def _py_lit(v) -> str:
    """Python-source-safe repr of a config value.

    json.dumps emits `true/false/null`, which aren't valid Python — `repr` is
    safer for the literals we use here (strings, lists, ints, floats, bools).
    """
    return repr(v)


def _path_or_none(rel: str | None, indent: int = 0) -> str:
    if rel is None:
        return "None"
    return f"PAIR_DIR / {_py_lit(rel)}"


def _params_cell(cfg: PairConfig) -> str:
    lines = [
        "# === PARAMETER CELL ===",
        "# Tweak paths and knobs here. Everything below should run unchanged.",
        "from pathlib import Path",
        "",
        "PAIR_DIR = Path('.').resolve()   # the dir containing this notebook",
        "",
        "# --- model checkpoints (None = side not yet trained / placeholder) ---",
        f"MODEL_OLD_PATH      = {_path_or_none(cfg.model_old_rel)}",
        f"MODEL_IMPROVED_PATH = {_path_or_none(cfg.model_improved_rel)}",
        "",
        "# --- encoded test pickles ---",
        f"TEST_PKL_OLD      = {_path_or_none(cfg.test_pkl_old_rel)}",
        f"TEST_PKL_IMPROVED = {_path_or_none(cfg.test_pkl_improved_rel)}",
        "",
        "# --- where the chunked sampling results land (also where batch_evaluate reads from) ---",
        f"SAMPLED_DIR_OLD      = PAIR_DIR / {_py_lit(cfg.sampled_old_rel)}",
        f"SAMPLED_DIR_IMPROVED = PAIR_DIR / {_py_lit(cfg.sampled_improved_rel)}",
        "",
        "# --- comparison output ---",
        f"COMPARISON_PKL = PAIR_DIR / {_py_lit(f'{cfg.dataset}_old_vs_improved.pkl')}",
        f"CAPTION        = {_py_lit(cfg.caption)}",
        "",
        "# --- sampling knobs (ProbabilisticEvaluation kwargs) ---",
        f"CONCEPT_NAME        = {_py_lit(cfg.concept_name)}",
        f"ALL_CAT             = {_py_lit(cfg.all_cat)}",
        f"ALL_NUM             = {_py_lit(cfg.all_num)}",
        f"GROWING_NUM_VALUES  = {_py_lit(cfg.growing_num_values)}",
        f"NUM_PROCESSES       = {_py_lit(cfg.num_processes)}",
        f"SAMPLES_PER_CASE    = {_py_lit(cfg.samples_per_case)}",
        f"SAVE_EVERY          = {_py_lit(cfg.save_every)}",
        f"RANDOM_ORDER        = {_py_lit(cfg.random_order)}",
        f"USE_VARIANCE_CAT    = True",
        f"USE_VARIANCE_NUM    = True",
        f"SAMPLE_ARGMAX       = False",
        "",
        "# --- metric knobs ---",
        f"ACTIVITY_KEY      = {_py_lit(cfg.activity_key)}",
        f"EVENT_LABEL_LIST  = {_py_lit(cfg.event_label_list)}",
        f"VALUE_FACTOR_TIME = {_py_lit(cfg.value_factor_time)}   # 3600*24 reports remaining-time in days",
    ]
    return "\n".join(lines) + "\n"


_IMPORTS_CELL = """\
import sys, importlib
from pathlib import Path

# Reach `src/` so `model.*`, `src.evaluation_metrics.*` resolve.
_REPO_ROOT = Path('.').resolve()
while not (_REPO_ROOT / 'src').is_dir() and _REPO_ROOT != _REPO_ROOT.parent:
    _REPO_ROOT = _REPO_ROOT.parent
for p in (str(_REPO_ROOT), str(_REPO_ROOT / 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reach the _shared helper module.
_SHARED = _REPO_ROOT / 'src' / 'interpretability' / 'improved_pipeline' / '_shared'
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

import comparison_helpers
importlib.reload(comparison_helpers)
from comparison_helpers import (
    SamplingConfig, ensure_sampled, default_metric_set, evaluate_dir,
    comparison_table, plot_overlay, save_results,
)
"""


_RUN_SAMPLE_CELL = """\
sampling_cfg = SamplingConfig(
    concept_name=CONCEPT_NAME,
    growing_num_values=GROWING_NUM_VALUES,
    all_cat=ALL_CAT,
    all_num=ALL_NUM,
    num_processes=NUM_PROCESSES,
    samples_per_case=SAMPLES_PER_CASE,
    sample_argmax=SAMPLE_ARGMAX,
    use_variance_cat=USE_VARIANCE_CAT,
    use_variance_num=USE_VARIANCE_NUM,
    random_order=RANDOM_ORDER,
    save_every=SAVE_EVERY,
)

if MODEL_OLD_PATH and TEST_PKL_OLD:
    ensure_sampled(MODEL_OLD_PATH, TEST_PKL_OLD, SAMPLED_DIR_OLD, sampling_cfg)
else:
    print('[skip] OLD side: missing MODEL_OLD_PATH or TEST_PKL_OLD')

if MODEL_IMPROVED_PATH and TEST_PKL_IMPROVED:
    ensure_sampled(MODEL_IMPROVED_PATH, TEST_PKL_IMPROVED, SAMPLED_DIR_IMPROVED, sampling_cfg)
else:
    print('[skip] IMPROVED side: missing MODEL_IMPROVED_PATH or TEST_PKL_IMPROVED')
"""


_BUILD_METRICS_CELL = """\
metrics = default_metric_set(
    activity_key=ACTIVITY_KEY,
    event_label_list=EVENT_LABEL_LIST,
    value_factor_time=VALUE_FACTOR_TIME,
)
print(f'metric set has {len(metrics)} entries')
"""


_EVAL_CELL = """\
res_old, counts_old = (None, None)
res_improved, counts_improved = (None, None)

if SAMPLED_DIR_OLD.is_dir() and any(SAMPLED_DIR_OLD.glob('results_part_*.pkl')):
    res_old, counts_old = evaluate_dir(SAMPLED_DIR_OLD, metrics)
else:
    print('[skip] OLD eval: no chunks under', SAMPLED_DIR_OLD)

if SAMPLED_DIR_IMPROVED.is_dir() and any(SAMPLED_DIR_IMPROVED.glob('results_part_*.pkl')):
    res_improved, counts_improved = evaluate_dir(SAMPLED_DIR_IMPROVED, metrics)
else:
    print('[skip] IMPROVED eval: no chunks under', SAMPLED_DIR_IMPROVED)
"""


_TABLE_CELL = """\
import pandas as pd
df = comparison_table(res_old, res_improved)
pd.set_option('display.max_rows', 200)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
df
"""


_PLOT_CELL = """\
plot_overlay(res_old, res_improved, counts_old, counts_improved, caption=CAPTION, pgf=False)
"""


_SAVE_CELL = """\
save_results(
    COMPARISON_PKL,
    res_old=res_old, counts_old=counts_old,
    res_improved=res_improved, counts_improved=counts_improved,
    config_old=sampling_cfg, config_improved=sampling_cfg,
)
"""


_FOOTER_MD = """\
## Notes

- The comparison pickle written in the last cell holds both `(res_raw, counts)` pairs and the sampling configs that produced them. Re-load it later with `comparison_helpers.load_results(path)`.
- To force re-sampling, pass `force=True` into the explicit `ensure_sampled` calls (or just delete the chunked output dir).
- For dataset-specific metric tweaks, copy `default_metric_set` into a cell and edit it; the rest of the pipeline only cares that `metrics` is a `dict[str, metric]`.
"""


def build_notebook(cfg: PairConfig) -> dict:
    cells = [
        _md(_intro_md(cfg)),
        _md("## Parameters"),
        _code(_params_cell(cfg)),
        _md("## Setup"),
        _code(_IMPORTS_CELL),
        _md("## MC suffix sampling (skipped per side if chunks already exist)"),
        _code(_RUN_SAMPLE_CELL),
        _md("## Build metric set"),
        _code(_BUILD_METRICS_CELL),
        _md("## Evaluate sampled outputs"),
        _code(_EVAL_CELL),
        _md("## Side-by-side metric table"),
        _code(_TABLE_CELL),
        _md("## Overlay plots"),
        _code(_PLOT_CELL),
        _md("## Save comparison pickle"),
        _code(_SAVE_CELL),
        _md(_FOOTER_MD),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    written = 0
    for cfg in PAIRS:
        pair_dir = PIPELINE_ROOT / cfg.author / cfg.dataset
        if not pair_dir.is_dir():
            print(f"[skip] {pair_dir} does not exist")
            continue
        nb_path = pair_dir / "old_vs_improved.ipynb"
        nb = build_notebook(cfg)
        with nb_path.open("w") as f:
            json.dump(nb, f, indent=1)
        print(f"wrote {nb_path}")
        written += 1
    print(f"\ndone: {written} notebooks written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
