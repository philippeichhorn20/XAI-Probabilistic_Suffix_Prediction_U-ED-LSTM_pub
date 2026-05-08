"""Shared helpers for the old-vs-improved comparison notebooks.

Each `improved_pipeline/<author>/<dataset>/old_vs_improved.ipynb` calls into
this module so the repeated logic (MC suffix sampling, batch-evaluating the
chunked outputs, side-by-side metric tables, overlaid interval plots) lives
in exactly one place.

The shape mirrors the existing pipeline:
- Sampling = `src/notebooks/evaluation_run_notebooks/<dataset>/<...>.ipynb`
- Metric evaluation = `src/notebooks/evaluation_metric_notebooks/improved/<dataset>/<...>.ipynb`

Sampling produces chunked `results_part_NNN.pkl` files in an output dir;
`batch_evaluate` ingests that dir and returns a `(results_dict, prefix/suffix
counts)` tuple. Re-running with sampled outputs already in place skips the
expensive MC step (`ensure_sampled` does the existence check).
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

# Allow callers to import without paying torch.jit warmup unless they use it.


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    """Knobs for the MC suffix-sampling step.

    Mirrors the kwargs passed to `ProbabilisticEvaluation` plus the chunked-save
    loop used in `evaluation_run_notebooks/`.
    """
    concept_name: str = "concept:name"
    eos_value: str = "EOS"
    growing_num_values: list[str] = field(default_factory=lambda: ["case_elapsed_time"])
    all_cat: list[str] | None = None
    all_num: list[str] | None = None
    num_processes: int = 16
    samples_per_case: int = 100
    sample_argmax: bool = False
    use_variance_cat: bool = True
    use_variance_num: bool = True
    variational_dropout_sampling: bool = False
    random_order: bool = False
    save_every: int = 100


def _has_sampled_chunks(output_dir: Path) -> bool:
    if not output_dir.is_dir():
        return False
    return any(p.name.startswith("results_part_") and p.suffix == ".pkl" for p in output_dir.iterdir())


def ensure_sampled(
    model_path: Path,
    test_dataset_path: Path,
    output_dir: Path,
    config: SamplingConfig,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Run MC suffix sampling into `output_dir` if no chunks are present.

    Skips work when `output_dir` already contains `results_part_*.pkl` files
    unless `force=True`.

    Returns `output_dir` so callers can chain into `batch_evaluate`.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not force and _has_sampled_chunks(output_dir):
        if verbose:
            n_chunks = sum(1 for _ in output_dir.glob("results_part_*.pkl"))
            print(f"[sample] {output_dir} already has {n_chunks} chunks — skipping (force=True to redo).")
        return output_dir

    # Lazy import: keeps the module light when only metric evaluation is wanted.
    from model.dropout_uncertainty_enc_dec_LSTM.dropout_uncertainty_model import (
        DropoutUncertaintyEncoderDecoderLSTM,
    )
    from src.evaluation.probabilistic_evaluation import ProbabilisticEvaluation

    if verbose:
        print(f"[sample] loading model from {model_path}")
    model = DropoutUncertaintyEncoderDecoderLSTM.load(str(model_path), dropout=0.0)
    model.eval()

    if verbose:
        print(f"[sample] loading test dataset from {test_dataset_path}")
    test_dataset = torch.load(str(test_dataset_path), weights_only=False)

    evaluator = ProbabilisticEvaluation(
        model,
        test_dataset,
        concept_name=config.concept_name,
        eos_value=config.eos_value,
        growing_num_values=list(config.growing_num_values),
        all_cat=list(config.all_cat) if config.all_cat is not None else None,
        all_num=list(config.all_num) if config.all_num is not None else None,
        num_processes=config.num_processes,
        samples_per_case=config.samples_per_case,
        sample_argmax=config.sample_argmax,
        use_variance_cat=config.use_variance_cat,
        use_variance_num=config.use_variance_num,
        variational_dropout_sampling=config.variational_dropout_sampling,
    )

    def save_chunk(buf: dict, last_i: int) -> None:
        chunk_number = last_i + 1
        path = output_dir / f"results_part_{chunk_number:03d}.pkl"
        with path.open("wb") as f:
            pickle.dump(buf, f)
        if verbose:
            print(f"[sample] wrote {len(buf)} prefixes -> {path.name}")

    buf: dict[tuple, tuple] = {}
    last_i = -1
    for i, (case_name, prefix_len, prefix, predicted_suffixes, suffix, mean_prediction) in enumerate(
        evaluator.evaluate_multi_processing(
            random_order=config.random_order, include_model_states=False
        )
    ):
        assert (case_name, prefix_len) not in buf
        buf[(case_name, prefix_len)] = (prefix, suffix, mean_prediction, predicted_suffixes)
        last_i = i
        if (i + 1) % config.save_every == 0:
            save_chunk(buf, i)
            buf = {}

    if buf:
        save_chunk(buf, last_i)

    if verbose:
        print(f"[sample] done. {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Metric evaluation
# ---------------------------------------------------------------------------

def default_metric_set(activity_key: str, event_label_list: list[str], *,
                       value_factor_time: float = 3600 * 24,
                       outlier_percentile: float = 0.25):
    """Build the metric dict used by Henryk's improved eval notebooks.

    `activity_key` is the dataset's concept-name column (e.g. 'Activity' for
    Helpdesk, 'concept:name' for BPIC17 / Sepsis). `value_factor_time` rescales
    the time targets — `3600*24` reports remaining-time in days.
    """
    from src.evaluation_metrics import metrics as M

    metrics: dict[str, Any] = {
        "NormalizedDamerauLevenshteinMeanVar_activity": M.NormalizedDamerauLevenshteinDistanceMeanVar(activity_key, percentile=outlier_percentile),
        "RemainingTime_SUM_Mean_MAE_outliers": M.SumValueMeanMAE("event_elapsed_time", percentile=outlier_percentile, value_factor=value_factor_time),
        "RemainingTime_SUM_Abs_Mean_MAE_outliers": M.SumAbsValueMeanMAE("event_elapsed_time", percentile=outlier_percentile, value_factor=value_factor_time),
        "RemainingTime_LastEvent_Mean2_Var_seconds_outliers": M.LastValueMean2VarMAE("case_elapsed_time", percentile=outlier_percentile, value_factor=value_factor_time),
        "RemainingTime_PIT": M.LastValuePIT("case_elapsed_time"),
        "EventElapsed_PIT": M.SumValuesPIT("event_elapsed_time"),
        "SuffixCountMAE": M.SuffixCountMAE(percentile=outlier_percentile),
    }
    for pct in (0.50, 0.75, 0.90, 0.95, 0.99):
        tag = f"{int(pct * 100)}"
        metrics[f"SumValuesInterval_{tag}"] = M.SumValuesInterval("event_elapsed_time", percentile=pct)
        metrics[f"LastValueInterval_{tag}"] = M.LastValueInterval("case_elapsed_time", percentile=pct)
        metrics[f"SuffixCountInterval_{tag}"] = M.SuffixCountInterval(percentile=pct)
        metrics[f"EventLabelCountInterval_{tag}"] = M.EventLabelCountInterval(activity_key, list(event_label_list), percentile=pct)
    return metrics


def evaluate_dir(output_dir: Path, metrics: dict, num_workers: int = 4) -> tuple[dict, tuple]:
    """Wrapper around `batch_evaluate` that pretty-prints the dir first.

    Returns `(res_raw, (prefix_count, suffix_count))` exactly like the upstream
    function, suitable for plumbing into `plots.plot_res(res, c, ...)`.
    """
    from src.evaluation_metrics.conduct_evaluation import batch_evaluate

    output_dir = Path(output_dir)
    if not _has_sampled_chunks(output_dir):
        raise FileNotFoundError(f"No results_part_*.pkl chunks under {output_dir}; run sampling first.")
    print(f"[eval] batch_evaluate({output_dir})")
    return batch_evaluate(str(output_dir), metrics, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Comparison summary + plotting
# ---------------------------------------------------------------------------

# Metrics where lower = better. Used to compute Δ% and color the deltas.
_LOWER_IS_BETTER = {
    "SuffixCountMAE",
    "NormalizedDamerauLevenshteinMeanVar_activity",
    "RemainingTime_SUM_Mean_MAE_outliers",
    "RemainingTime_SUM_Abs_Mean_MAE_outliers",
    "RemainingTime_LastEvent_Mean2_Var_seconds_outliers",
}


def _agg_metric(metric_results: dict, kind: str) -> float | None:
    """Aggregate a per-(case,prefix,suffix) metric dict into one number.

    `kind='mean'` averages the deterministic mean prediction's score across
    cases; `kind='prob'` averages the probabilistic score (samples-based).
    Returns None if the metric isn't aggregateable in that mode.
    """
    if not metric_results:
        return None
    vals: list[float] = []
    for v in metric_results.values():
        if not isinstance(v, dict):
            return None
        if kind == "mean" and "mean" in v:
            vals.append(float(v["mean"]))
        elif kind == "prob" and "prob" in v:
            p = v["prob"]
            vals.append(float(p[0]) if isinstance(p, (list, tuple)) and p else float("nan"))
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else None


def comparison_table(
    res_old: dict | None,
    res_improved: dict | None,
    metric_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build a side-by-side table from two `res_raw` dicts.

    Either side may be None (e.g. when the improved checkpoint isn't trained
    yet) — those columns will be NaN. Returns one row per (metric, kind)
    where kind ∈ {mean, prob}.
    """
    if metric_keys is None:
        keys = sorted(set((res_old or {}).keys()) | set((res_improved or {}).keys()))
    else:
        keys = list(metric_keys)
    rows = []
    for key in keys:
        for kind in ("mean", "prob"):
            v_old = _agg_metric((res_old or {}).get(key, {}), kind)
            v_imp = _agg_metric((res_improved or {}).get(key, {}), kind)
            delta = None
            delta_pct = None
            if v_old is not None and v_imp is not None and v_old != 0:
                delta = v_imp - v_old
                delta_pct = (v_imp - v_old) / abs(v_old) * 100
            rows.append({
                "metric": key,
                "kind": kind,
                "old": v_old,
                "improved": v_imp,
                "delta": delta,
                "delta_pct": delta_pct,
                "lower_is_better": key in _LOWER_IS_BETTER,
            })
    df = pd.DataFrame(rows, columns=["metric", "kind", "old", "improved", "delta", "delta_pct", "lower_is_better"])
    if not df.empty:
        df = df.dropna(subset=["old", "improved"], how="all").reset_index(drop=True)
    return df


def plot_overlay(
    res_old: dict | None,
    res_improved: dict | None,
    counts_old: tuple,
    counts_improved: tuple,
    *,
    caption: str = "",
    pgf: bool = False,
):
    """Overlay the standard interval-plot stack for old vs improved.

    Falls back to plotting only the side that's available when one is missing.
    Uses the existing `plot_res` so figure layout matches Henryk's notebooks.
    """
    from src.evaluation_metrics import plots

    keep = [
        ("SuffixCountMAE", "Suffix length MAE"),
        ("NormalizedDamerauLevenshteinMeanVar_activity", "DLS"),
        ("RemainingTime_SUM_Mean_MAE_outliers", "Rem. time (event sum) MAE (days)"),
        ("SuffixCountInterval_50", "SuffixCountInterval_50"),
        ("SuffixCountInterval_75", "SuffixCountInterval_75"),
        ("SuffixCountInterval_90", "SuffixCountInterval_90"),
        ("SuffixCountInterval_95", "SuffixCountInterval_95"),
        ("SuffixCountInterval_99", "SuffixCountInterval_99"),
        ("SumValuesInterval_50", "SumValuesInterval_50"),
        ("SumValuesInterval_75", "SumValuesInterval_75"),
        ("SumValuesInterval_90", "SumValuesInterval_90"),
        ("SumValuesInterval_95", "SumValuesInterval_95"),
        ("SumValuesInterval_99", "SumValuesInterval_99"),
        ("LastValueInterval_50", "LastValueInterval_50"),
        ("LastValueInterval_75", "LastValueInterval_75"),
        ("LastValueInterval_90", "LastValueInterval_90"),
        ("LastValueInterval_95", "LastValueInterval_95"),
        ("LastValueInterval_99", "LastValueInterval_99"),
    ]
    for label, res, counts in (("OLD", res_old, counts_old), ("IMPROVED", res_improved, counts_improved)):
        if res is None or not counts:
            continue
        sub = {(k, l): res[k] for k, l in keep if k in res}
        if not sub:
            print(f"[plot] skipping {label}: no overlapping metric keys")
            continue
        print(f"[plot] {label}: {caption}")
        plots.plot_res(sub, counts, columns=2, caption=f"{caption} — {label}", pgf=pgf)


def save_results(
    path: Path,
    res_old,
    counts_old,
    res_improved,
    counts_improved,
    *,
    config_old: SamplingConfig | None = None,
    config_improved: SamplingConfig | None = None,
) -> None:
    """Pickle the two `(res, counts)` pairs + the sampling configs that produced them."""
    payload = {
        "old": (res_old, counts_old),
        "improved": (res_improved, counts_improved),
        "config_old": config_old,
        "config_improved": config_improved,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"[save] wrote {path}")


def load_results(path: Path) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)
