"""Camargo-flavoured suffix sampling for old_vs_improved notebooks.

Camargo (`SharedCat_LSTM` / `FullShared_Join_LSTM`) is a deterministic
single-step next-activity predictor. It has no time / side-channel output, so
the U-ED-LSTM-shaped MC-sampling helpers in `comparison_helpers.py` don't
apply. This module provides:

- ``ensure_sampled_camargo``: argmax + stochastic-softmax suffix decoding,
  written in the same chunked tuple format that ``evaluate_dir`` consumes:
  ``buf[(case_id, prefix_len)] = (prefix, suffix, mean_prediction, predicted_suffixes)``.
  Each event-dict carries only the activity feature (e.g. ``concept:name``) —
  time metrics are not meaningful for Camargo and should be omitted.

- ``default_metric_set_camargo``: activity-only metric subset
  (DamerauLevenshtein, SuffixCountMAE, EventLabelCountInterval). Drops every
  metric that touches ``case_elapsed_time`` / ``event_elapsed_time``.

The intent is that an `old_vs_improved` notebook can call ``ensure_sampled``
for the henryk side and ``ensure_sampled_camargo`` for the camargo side; the
metric/plot machinery downstream is unchanged.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from comparison_helpers import SamplingConfig, _has_sampled_chunks


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

@dataclass
class CamargoSpec:
    """Minimum spec needed by the Camargo sampler.

    Mirrors the relevant fields of ``_camargo_utils.DatasetSpec`` so callers
    can build one inline without importing the full notebook helper.
    """
    model_class: str               # 'SharedCat_LSTM' or 'FullShared_Join_LSTM'
    cat_indices: tuple[int, ...]   # categorical feature positions consumed by the model
    num_indices: tuple[int, ...]   # numerical feature positions
    activity_feature: str          # e.g. 'concept:name' or 'Activity'
    ngram_size: int                # paper n-gram window. 0 = use loader's compute_window


def _load_camargo_model(model_path: Path, model_class: str):
    """Lazy-load the right Camargo model class.

    Both classes live under ``src/reimplemented_comparable_approaches/
    camargo_LSTM_suffix_pred/`` and ship a classmethod ``.load(path)``.
    """
    if model_class == "SharedCat_LSTM":
        from sharedCatLSTM.model import SharedCat_LSTM  # type: ignore
        return SharedCat_LSTM.load(str(model_path))
    if model_class == "FullShared_Join_LSTM":
        from joinLSTM.model import FullShared_Join_LSTM  # type: ignore
        return FullShared_Join_LSTM.load(str(model_path))
    raise ValueError(f"Unknown Camargo model class: {model_class!r}")


def _make_initial_window(cat_t: tuple[torch.Tensor, ...],
                         num_t: tuple[torch.Tensor, ...],
                         start: int,
                         prefix_len: int,
                         window: int,
                         spec: CamargoSpec) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build pad-left window tensors of shape [1, window] for the requested
    prefix. Mirrors `_camargo_utils.make_prefix_window` but returns lists
    aligned with ``spec.cat_indices`` / ``spec.num_indices`` (no projection
    helper dependency)."""
    cats_sub = [cat_t[i] for i in spec.cat_indices]
    nums_sub = [num_t[i] for i in spec.num_indices]
    cats_out = [torch.zeros((1, window), dtype=c.dtype) for c in cats_sub]
    nums_out = [torch.zeros((1, window), dtype=n.dtype) for n in nums_sub]
    effective = min(prefix_len, window)
    src_offset = prefix_len - effective
    for k in range(effective):
        dst = window - effective + k
        src = start + src_offset + k
        for j in range(len(cats_sub)):
            cats_out[j][0, dst] = cats_sub[j][src]
        for j in range(len(nums_sub)):
            nums_out[j][0, dst] = nums_sub[j][src]
    return cats_out, nums_out


def _shift_and_append(cats: list[torch.Tensor], nums: list[torch.Tensor],
                      next_activity: int) -> None:
    """Roll the window one step left in-place and append the predicted
    activity at the rightmost slot. Side-channel categoricals and numericals
    carry forward (last observed value), since SharedCat doesn't predict
    them. This matches what the Camargo paper does at inference time.

    Convention: cat[0] is the activity (the model puts it first per
    `model_feat`). All other features are carried forward from the previous
    rightmost slot."""
    # Shift left by one position.
    for c in cats:
        c[0, :-1] = c[0, 1:].clone()
    for n in nums:
        n[0, :-1] = n[0, 1:].clone()
    # Append at the rightmost slot. cat[0] = predicted activity; other cats
    # and nums keep whatever value was most recently in slot [-2] (now [-1]
    # after the shift), i.e. last-observed carry forward.
    cats[0][0, -1] = next_activity


def _sample_suffix(model,
                   cats: list[torch.Tensor],
                   nums: list[torch.Tensor],
                   eos_idx: int,
                   max_steps: int,
                   inv_vocab: dict[int, str],
                   activity_feature: str,
                   *,
                   stochastic: bool,
                   temperature: float = 1.0) -> list[dict]:
    """Auto-regressively decode a suffix from the model, stopping at EOS or
    ``max_steps``. Returns a list of dicts ``[{activity_feature: name}, ...]``
    (no time / side-channel fields)."""
    suffix: list[dict] = []
    cats = [c.clone() for c in cats]
    nums = [n.clone() for n in nums]
    with torch.no_grad():
        for _ in range(max_steps):
            logits = model((list(cats), list(nums)))
            if stochastic and temperature > 0:
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                # Mask padding (idx 0) so it's never sampled.
                probs = probs.clone()
                probs[..., 0] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_idx = int(torch.multinomial(probs[0], num_samples=1).item())
            else:
                # Argmax over non-padding entries.
                logits_clone = logits.clone()
                logits_clone[..., 0] = float('-inf')
                next_idx = int(logits_clone.argmax(dim=-1).item())
            if next_idx == eos_idx:
                break
            suffix.append({activity_feature: inv_vocab.get(next_idx, f'<unk:{next_idx}>')})
            _shift_and_append(cats, nums, next_idx)
    return suffix


def _readable_prefix(cat_t: tuple[torch.Tensor, ...],
                     start: int,
                     prefix_len: int,
                     activity_idx_in_dataset: int,
                     inv_vocab: dict[int, str],
                     activity_feature: str) -> list[dict]:
    """Decode the prefix as activity-only readable dicts. Used by metrics
    that index into ``prefix[-1]`` (e.g. boundary metrics) — those metrics
    aren't part of the camargo metric set, but build it anyway so the tuple
    shape matches henryk-side outputs."""
    out: list[dict] = []
    for k in range(prefix_len):
        idx = int(cat_t[activity_idx_in_dataset][start + k].item())
        if idx == 0:
            continue
        out.append({activity_feature: inv_vocab.get(idx, f'<unk:{idx}>')})
    return out


def _readable_suffix(cat_t: tuple[torch.Tensor, ...],
                     start: int,
                     prefix_len: int,
                     useful_len: int,
                     activity_idx_in_dataset: int,
                     inv_vocab: dict[int, str],
                     activity_feature: str,
                     eos_idx: int) -> list[dict]:
    out: list[dict] = []
    for k in range(prefix_len, useful_len):
        idx = int(cat_t[activity_idx_in_dataset][start + k].item())
        if idx == 0 or idx == eos_idx:
            break
        out.append({activity_feature: inv_vocab.get(idx, f'<unk:{idx}>')})
    return out


def ensure_sampled_camargo(
    model_path: Path,
    test_dataset_path: Path,
    output_dir: Path,
    config: SamplingConfig,
    *,
    spec: CamargoSpec,
    sampling_temperature: float = 1.0,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Run Camargo-style suffix sampling and write chunked outputs.

    If the ``output_dir`` already contains ``results_part_*.pkl`` files and
    ``force=False``, we skip — same contract as ``ensure_sampled``.

    For each (case, prefix_len), produces ``samples_per_case`` stochastic
    softmax-sampled suffixes plus one argmax ``mean_prediction``. Every
    suffix step is a dict with a single key (``spec.activity_feature``).
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not force and _has_sampled_chunks(output_dir):
        if verbose:
            n = sum(1 for _ in output_dir.glob("results_part_*.pkl"))
            print(f"[sample-camargo] {output_dir} already has {n} chunks — skipping (force=True to redo).")
        return output_dir

    if verbose:
        print(f"[sample-camargo] loading {spec.model_class} from {model_path}")
    model = _load_camargo_model(model_path, spec.model_class)
    model.eval()

    if verbose:
        print(f"[sample-camargo] loading test dataset from {test_dataset_path}")
    test_dataset = torch.load(str(test_dataset_path), weights_only=False)

    all_cats = test_dataset.all_categories[0]
    activity_idx_in_dataset = next(
        i for i, c in enumerate(all_cats) if c[0] == spec.activity_feature
    )
    activity_vocab = all_cats[activity_idx_in_dataset][2]
    inv_vocab = {v: k for k, v in activity_vocab.items()}
    if "EOS" not in activity_vocab:
        raise ValueError(
            f"Activity feature '{spec.activity_feature}' has no EOS token in vocab; "
            "cannot decode suffixes."
        )
    eos_idx = activity_vocab["EOS"]

    seq_len = test_dataset[0][0][0].shape[0]
    window = spec.ngram_size if spec.ngram_size > 0 else seq_len
    max_decode_steps = seq_len  # safe upper bound

    # Build the per-case prefix iteration: longest unique trace per case_id.
    case_to_best: dict[Any, tuple[int, int]] = {}  # case_id -> (ds_idx, useful_len)
    for ds_idx in range(len(test_dataset)):
        cat_t, _, case_id = test_dataset[ds_idx]
        # Find first non-padding and trailing EOS / padding to compute useful length.
        acts = cat_t[activity_idx_in_dataset]
        nonzero = (acts != 0).nonzero(as_tuple=True)[0]
        if len(nonzero) == 0:
            continue
        start = int(nonzero[0].item())
        useful_len = int(len(nonzero))
        # strip trailing EOS
        while useful_len > 0 and int(acts[start + useful_len - 1].item()) == eos_idx:
            useful_len -= 1
        if case_id not in case_to_best or useful_len > case_to_best[case_id][1]:
            case_to_best[case_id] = (ds_idx, useful_len)

    buf: dict[tuple, tuple] = {}
    last_global_i = -1

    def save_chunk(buf_to_save: dict[tuple, tuple], last_i: int) -> None:
        chunk_number = last_i + 1
        path = output_dir / f"results_part_{chunk_number:03d}.pkl"
        with path.open("wb") as f:
            pickle.dump(buf_to_save, f)
        if verbose:
            print(f"[sample-camargo] wrote {len(buf_to_save)} prefixes -> {path.name}")

    n_total_prefixes = sum(max(0, ul - 1) for _, ul in case_to_best.values())
    pbar = tqdm(total=n_total_prefixes, disable=not verbose,
                desc=f"camargo-sampling ({config.samples_per_case}/case)")

    global_i = -1
    for case_id, (ds_idx, useful_len) in case_to_best.items():
        cat_t, num_t, _ = test_dataset[ds_idx]
        # Find start (offset of first non-zero activity in the padded tensor).
        acts = cat_t[activity_idx_in_dataset]
        start = int((acts != 0).nonzero(as_tuple=True)[0][0].item())
        for prefix_len in range(1, useful_len):
            global_i += 1
            cats_w, nums_w = _make_initial_window(cat_t, num_t, start, prefix_len, window, spec)
            mean_prediction = _sample_suffix(
                model, cats_w, nums_w, eos_idx, max_decode_steps,
                inv_vocab, spec.activity_feature, stochastic=False,
            )
            predicted_suffixes = []
            for _ in range(config.samples_per_case):
                predicted_suffixes.append(_sample_suffix(
                    model, cats_w, nums_w, eos_idx, max_decode_steps,
                    inv_vocab, spec.activity_feature, stochastic=True,
                    temperature=sampling_temperature,
                ))
            prefix_readable = _readable_prefix(
                cat_t, start, prefix_len,
                activity_idx_in_dataset, inv_vocab, spec.activity_feature,
            )
            suffix_readable = _readable_suffix(
                cat_t, start, prefix_len, useful_len,
                activity_idx_in_dataset, inv_vocab, spec.activity_feature,
                eos_idx,
            )
            buf[(case_id, prefix_len)] = (
                prefix_readable, suffix_readable, mean_prediction, predicted_suffixes,
            )
            last_global_i = global_i
            pbar.update(1)
            if (global_i + 1) % config.save_every == 0:
                save_chunk(buf, global_i)
                buf = {}
    pbar.close()

    if buf:
        save_chunk(buf, last_global_i)

    if verbose:
        print(f"[sample-camargo] done. {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Activity-only metric set
# ---------------------------------------------------------------------------

def default_metric_set_camargo(activity_key: str, event_label_list: list[str], *,
                                outlier_percentile: float = 0.25):
    """Activity-only metric subset for Camargo.

    Drops every time-based metric in ``default_metric_set`` because Camargo
    doesn't predict ``case_elapsed_time`` / ``event_elapsed_time``.
    Includes:

    - ``NormalizedDamerauLevenshteinMeanVar_activity`` — suffix DL distance.
    - ``SuffixCountMAE`` — suffix-length error.
    - ``EventLabelCountInterval_*`` — count distribution over named labels.
    """
    from src.evaluation_metrics import metrics as M

    metrics: dict[str, Any] = {
        "NormalizedDamerauLevenshteinMeanVar_activity":
            M.NormalizedDamerauLevenshteinDistanceMeanVar(activity_key, percentile=outlier_percentile),
        "SuffixCountMAE": M.SuffixCountMAE(percentile=outlier_percentile),
    }
    for pct in (0.50, 0.75, 0.90, 0.95, 0.99):
        tag = f"{int(pct * 100)}"
        metrics[f"SuffixCountInterval_{tag}"] = M.SuffixCountInterval(percentile=pct)
        metrics[f"EventLabelCountInterval_{tag}"] = M.EventLabelCountInterval(
            activity_key, list(event_label_list), percentile=pct,
        )
    return metrics
