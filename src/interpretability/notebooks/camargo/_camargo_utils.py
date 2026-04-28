"""Shared utilities for the Camargo notebooks.

The Camargo `FullShared_Join_LSTM` is a single-step next-activity predictor
with embeddings for two categorical features + concatenated numericals. It
takes fixed-length window tensors (pad-left convention: real events occupy
the last `prefix_length` slots; earlier slots are zeros).

Helpers in this module:

- `load_camargo_model`: load the checkpoint and put it in eval mode.
- `load_test_dataset`: load a henryk-style encoded pickle.
- `project_features`: extract only the feature tensors the Camargo model
  expects (cats: [activity_like, resource_like], nums: [case_elapsed_time])
  from a henryk dataset sample.
- `CamargoPredictor`: thin wrapper that accepts full-length case tensors,
  produces a [batch, n_classes] softmax for the next-activity prediction at
  a given prefix length, with argmax convenience.
- `collect_prefix_predictions`: iterate the test set and gather
  (case, prefix_length, current_activity, actual_next, predicted_next, probs)
  tuples suitable for process-map / gateway analyses.

The Camargo and henryk datasets were produced with different preprocessing
runs, so the `case_elapsed_time` scaler differs slightly. Categorical
signals dominate behaviour so single-event prefixes are noisy but deeper
prefixes match what the Camargo evaluation notebooks produce.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import torch


def _ensure_on_path(project_root: Path) -> None:
    """Ensure both the project root and `src/` + Camargo source dir are on sys.path."""
    paths = [
        project_root,
        project_root / "src",
        project_root / "src" / "reimplemented_comparable_approaches" / "camargo_LSTM_suffix_pred",
    ]
    for p in paths:
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards until we see the `src/` directory."""
    p = (start or Path().resolve()).resolve()
    while p != p.parent:
        if (p / "src").is_dir() and (p / "encoded_data").is_dir():
            return p
        p = p.parent
    raise FileNotFoundError("Could not find project root (looking for src/ + encoded_data/).")


# Per-dataset knobs. Feature indices map the henryk dataset columns onto what
# the Camargo model expects: two categorical features (activity + resource-like)
# and one numerical feature (case_elapsed_time).
@dataclass(frozen=True)
class DatasetSpec:
    name: str                       # "Helpdesk" / "BPIC17"
    test_pickle: str                # relative to project root
    model_pickle: str               # relative to project root
    cat_indices: Tuple[int, int]    # (activity, resource) positions in henryk dataset cats
    num_indices: Tuple[int, ...]    # (case_elapsed_time,) position in nums
    activity_feature: str           # 'Activity' / 'concept:name'
    model_class: str = "FullShared_Join_LSTM"   # or "SharedCat_LSTM"
    ngram_size: int = 0             # >0 = Camargo n-gram window (paper §3.1); 0 = legacy compute_window


HELPDESK = DatasetSpec(
    name="Helpdesk",
    # Use the Camargo-preprocessed test set (compare_camargo/). This matches the
    # scaler + train/test split the Camargo model was trained with.
    test_pickle="encoded_data/compare_camargo/helpdesk_all_5_test.pkl",
    model_pickle=(
        "src/reimplemented_comparable_approaches/camargo_LSTM_suffix_pred/"
        "notebooks/training/Helpdesk/Helpdesk_camargo_act_1_suffix_length5.pkl"
    ),
    cat_indices=(0, 1),   # Activity, Resource
    num_indices=(0,),     # case_elapsed_time
    activity_feature="Activity",
)

HELPDESK_SHAREDCAT_ROLES = DatasetSpec(
    name="Helpdesk (SharedCat + Roles)",
    test_pickle="encoded_data/compare_camargo/helpdesk_all_5_roles_test.pkl",
    model_pickle=(
        "src/reimplemented_comparable_approaches/camargo_LSTM_suffix_pred/"
        "notebooks/training/Helpdesk/Helpdesk_camargo_sharedcat_roles_act_1_suffix_length5.pkl"
    ),
    cat_indices=(0, 1),   # Activity, Role
    num_indices=(0,),
    activity_feature="Activity",
    model_class="SharedCat_LSTM",
)


# Paper-faithful variant: per-time-step n-gram loader (Camargo §3.1), ngram_size=5 per Table 3.
HELPDESK_SHAREDCAT_ROLES_NGRAM = DatasetSpec(
    name="Helpdesk",
    test_pickle="encoded_data/compare_camargo/helpdesk_all_5_roles_test.pkl",
    model_pickle=(
        "src/reimplemented_comparable_approaches/camargo_LSTM_suffix_pred/"
        "notebooks/training/Helpdesk/Helpdesk_camargo_sharedcat_roles_ngram5.pkl"
    ),
    cat_indices=(0, 1),
    num_indices=(0,),
    activity_feature="Activity",
    model_class="SharedCat_LSTM",
    ngram_size=5,
)


BPIC17 = DatasetSpec(
    name="BPIC17",
    test_pickle="encoded_data/BPIC_2017_all_5_test.pkl",
    model_pickle=(
        "src/reimplemented_comparable_approaches/camargo_LSTM_suffix_pred/"
        "notebooks/training/BPIC17/BPIC17_camargo_act_1_suffix_length5.pkl"
    ),
    cat_indices=(0, 2),   # concept:name, org:resource
    num_indices=(0,),     # case_elapsed_time
    activity_feature="concept:name",
)


# Paper-faithful variant: per-time-step n-gram loader. ngram_size=15 chosen by
# analogy with BPI 2012 in Camargo Table 3 (similar complexity; no BPIC17 row).
BPIC17_NGRAM = DatasetSpec(
    name="BPIC17 (n-gram=15)",
    test_pickle="encoded_data/BPIC_2017_all_5_test.pkl",
    model_pickle=(
        "src/reimplemented_comparable_approaches/camargo_LSTM_suffix_pred/"
        "notebooks/training/BPIC17/BPIC17_camargo_ngram15.pkl"
    ),
    cat_indices=(0, 2),
    num_indices=(0,),
    activity_feature="concept:name",
    ngram_size=15,
)


def load_camargo_model(project_root: Path, spec: DatasetSpec):
    _ensure_on_path(project_root)
    if spec.model_class == "SharedCat_LSTM":
        from sharedCatLSTM.model import SharedCat_LSTM  # type: ignore
        model = SharedCat_LSTM.load(str(project_root / spec.model_pickle))
    else:
        from joinLSTM.model import FullShared_Join_LSTM  # type: ignore
        model = FullShared_Join_LSTM.load(str(project_root / spec.model_pickle))
    model.eval()
    return model


def load_test_dataset(project_root: Path, spec: DatasetSpec):
    _ensure_on_path(project_root)
    return torch.load(str(project_root / spec.test_pickle), weights_only=False)


def activity_vocab(model) -> Tuple[dict, dict, int]:
    """Return (name->idx, idx->name, eos_idx) for the model's activity head."""
    name_to_idx = model.data_set_categories[0][0][2]
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    eos_idx = name_to_idx["EOS"]
    return name_to_idx, idx_to_name, eos_idx


def project_features(cats_full: Sequence[torch.Tensor],
                     nums_full: Sequence[torch.Tensor],
                     spec: DatasetSpec) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    cats = [cats_full[i] for i in spec.cat_indices]
    nums = [nums_full[i] for i in spec.num_indices]
    return cats, nums


def make_prefix_window(cats_full: Sequence[torch.Tensor],
                       nums_full: Sequence[torch.Tensor],
                       start: int,
                       prefix_length: int,
                       window: int,
                       spec: DatasetSpec) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Build pad-left window tensors with the last `min(prefix_length, window)`
    real events of a trace (starting at `start`) right-aligned in a window of
    length `window`. When prefix_length > window, the oldest events are dropped
    — same truncation rule the Camargo n-gram trainer applies.

    Returns batched tensors of shape [1, window] each.
    """
    cats_sub, nums_sub = project_features(cats_full, nums_full, spec)
    cats_out = [torch.zeros((1, window), dtype=c.dtype) for c in cats_sub]
    nums_out = [torch.zeros((1, window), dtype=n.dtype) for n in nums_sub]
    effective = min(prefix_length, window)
    src_offset = prefix_length - effective  # skip oldest events when history exceeds window
    for k in range(effective):
        dst = window - effective + k
        src = start + src_offset + k
        for j in range(len(cats_sub)):
            cats_out[j][0, dst] = cats_sub[j][src]
        for j in range(len(nums_sub)):
            nums_out[j][0, dst] = nums_sub[j][src]
    return cats_out, nums_out


class CamargoPredictor:
    """Thin wrapper around FullShared_Join_LSTM with a predict/predict_batch API."""

    def __init__(self, model, spec: DatasetSpec, window: int):
        self.model = model
        self.spec = spec
        self.window = window
        self.name_to_idx, self.idx_to_name, self.eos_idx = activity_vocab(model)
        self.n_classes = len(model.data_set_categories[0][0][2]) + 1  # +1 for padding idx 0

    @torch.no_grad()
    def predict_distribution(self,
                             cats: Sequence[torch.Tensor],
                             nums: Sequence[torch.Tensor]) -> torch.Tensor:
        """cats/nums: lists of [B, window] tensors. Returns [B, n_classes] softmax."""
        logits = self.model((list(cats), list(nums)))
        return torch.nn.functional.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_batch(self,
                      cats: Sequence[torch.Tensor],
                      nums: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (argmax indices [B], probs [B, n_classes])."""
        probs = self.predict_distribution(cats, nums)
        return probs.argmax(dim=-1), probs


def iter_case_prefixes(cats_full: Sequence[torch.Tensor],
                       acts_tensor_idx: int,
                       eos_idx: int,
                       include_end: bool = False) -> Iterator[Tuple[int, int, int]]:
    """Yield (prefix_length, start, useful_len) for each position in a case with
    a real next activity to predict. `start` is the index of the first real
    event; `useful_len` is the number of real non-EOS events in the case.

    With include_end=True, also yields the prefix that ends at the last real
    activity (next event is EOS) so case-end transitions are captured."""
    acts = cats_full[acts_tensor_idx].tolist()
    # find first non-padding (idx 0 is padding) and the trailing real (non-EOS) run
    nonzero = [i for i, a in enumerate(acts) if a != 0]
    if not nonzero:
        return
    start = nonzero[0]
    useful_len = len(nonzero)
    # strip trailing EOS
    while useful_len > 0 and acts[start + useful_len - 1] == eos_idx:
        useful_len -= 1
    upper = useful_len if include_end else useful_len - 1
    for k in range(upper):
        prefix_length = k + 1
        yield prefix_length, start, useful_len


def collect_prefix_predictions(predictor: CamargoPredictor,
                               dataset,
                               progress: bool = True,
                               include_end: bool = False) -> List[dict]:
    """Collect one record per (case, prefix_length) with actual/predicted/probs.

    Records: {case_id, prefix_length, current_activity, actual_next_activity,
              predicted_next_activity, confidence, correct}

    With include_end=True, also records (last_real_activity -> EOS) so case-end
    predictions show up downstream (process map, surrogate gateway data, etc.).
    """
    model_act_idx = 0  # Camargo activity is always cat[0] after projection
    eos_idx = predictor.eos_idx
    idx_to_name = predictor.idx_to_name
    records = []

    # Unique cases: keep the longest trace per case_id
    cases = {}
    for i in range(len(dataset)):
        cat_t, _, cid = dataset[i]
        trace_len = int((cat_t[0] != 0).sum().item())
        if cid not in cases or trace_len > cases[cid][1]:
            cases[cid] = (i, trace_len)

    iterator = cases.items()
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, total=len(cases), desc=f"Collecting predictions ({predictor.spec.name})")
        except ImportError:
            pass

    for cid, (idx, _tl) in iterator:
        cats_full, nums_full, _ = dataset[idx]
        cats_proj, nums_proj = project_features(cats_full, nums_full, predictor.spec)
        acts_list = cats_proj[0].tolist()

        for prefix_length, start, useful_len in iter_case_prefixes(
            [cats_proj[0]], acts_tensor_idx=0, eos_idx=eos_idx,
            include_end=include_end,
        ):
            # Build prefix window
            cats_win, nums_win = make_prefix_window(
                cats_full, nums_full, start=start,
                prefix_length=prefix_length,
                window=predictor.window, spec=predictor.spec,
            )
            pred_idx, probs = predictor.predict_batch(cats_win, nums_win)
            pred_idx = int(pred_idx.item())
            current_activity = idx_to_name.get(acts_list[start + prefix_length - 1])
            actual_next_idx = acts_list[start + prefix_length]
            actual_next_activity = idx_to_name.get(actual_next_idx)
            predicted_next_activity = idx_to_name.get(pred_idx, f"<{pred_idx}>")
            records.append({
                "case_id": cid,
                "prefix_length": prefix_length,
                "current_activity": current_activity,
                "actual_next_activity": actual_next_activity,
                "predicted_next_activity": predicted_next_activity,
                "confidence": float(probs[0, pred_idx].item()),
                "correct": pred_idx == actual_next_idx,
                "probs": probs[0].cpu().numpy(),
            })

    return records


def gateway_dfg(records: List[dict]) -> dict:
    """From collected prefix predictions, build (current -> {next: count}) DFG
    over the ACTUAL transitions (ground truth next activity)."""
    dfg: dict = {}
    for r in records:
        dfg.setdefault(r["current_activity"], {}).setdefault(r["actual_next_activity"], 0)
        dfg[r["current_activity"]][r["actual_next_activity"]] += 1
    return dfg


def compute_window(dataset) -> int:
    """Return the prefix-window size (henryk sequence length minus min suffix size)."""
    total = dataset.encoder_decoder.window_size
    min_sfx = dataset.encoder_decoder.min_suffix_size
    return total - min_sfx


def _forward_from_embeddings(model, embedded_cats: torch.Tensor, nums_stacked: torch.Tensor) -> torch.Tensor:
    """Run the Camargo model starting from already-embedded continuous inputs.

    Two architectures are supported:
      - Join LSTM (BPIC17 default): shared_lstm takes [cats | nums]; head takes y.
      - SharedCat LSTM (Helpdesk SharedCat-Roles): shared_lstm takes cats only;
        nums are concatenated into the head input.

    The two are distinguished by shared_lstm.input_size: if it equals the cat-
    embedding dim, this is the cat-only variant.

    Args:
        embedded_cats: [B, T, cat_embed_total] concatenated embeddings for cat features.
        nums_stacked:  [B, T, n_num] stacked numerical inputs.

    Returns softmax probabilities [B, n_classes].
    """
    cat_only_shared = model.shared_lstm.input_size == embedded_cats.shape[-1]
    has_nums = nums_stacked.shape[-1] > 0

    if cat_only_shared:
        shared_in = embedded_cats
    else:
        shared_in = torch.cat([embedded_cats, nums_stacked], dim=-1) if has_nums else embedded_cats

    out_seq, _ = model.shared_lstm(shared_in)  # [B, T, H]
    y = model.bn1(out_seq.transpose(1, 2)).transpose(1, 2)  # [B, T, H]

    # SharedCat injects nums at the head; Join LSTM does not.
    head_in = torch.cat([y, nums_stacked], dim=-1) if (cat_only_shared and has_nums) else y

    _, (h_act, _) = model.lstm_act(head_in)
    a_logits = model.act_head(h_act.squeeze(0))
    return torch.nn.functional.softmax(a_logits, dim=-1)


def compute_integrated_gradients(model,
                                 cats_window: Sequence[torch.Tensor],
                                 nums_window: Sequence[torch.Tensor],
                                 target_class: int,
                                 n_steps: int = 50) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Integrated Gradients on the Camargo model at the embedding + numeric layer.

    Returns two lists of attribution tensors:
      - cat_attrs: one [T, embed_dim] tensor per categorical feature (summed over
        embedding dims gives per-position contribution).
      - num_attrs: one [T] tensor per numerical feature.

    The baseline uses zero-vectors for both embeddings and numerical inputs.
    """
    from captum.attr import IntegratedGradients  # type: ignore

    model.eval()
    # Embed the categorical features (batch dim = 1 by convention)
    embedded_per_cat = []
    for j, emb_layer in enumerate(model.embeddings):
        embedded_per_cat.append(emb_layer(cats_window[j]))  # [1, T, d_j]
    embedded_cats = torch.cat(embedded_per_cat, dim=-1)  # [1, T, sum_d]
    nums_stacked = (torch.stack([n.float() for n in nums_window], dim=-1)
                    if len(nums_window) else torch.zeros(embedded_cats.shape[0], embedded_cats.shape[1], 0))
    if nums_stacked.dim() == 2:
        nums_stacked = nums_stacked.unsqueeze(0)

    def forward_for_ig(emb_cats, nums):
        return _forward_from_embeddings(model, emb_cats, nums)

    ig = IntegratedGradients(forward_for_ig)
    cat_attr, num_attr = ig.attribute(
        inputs=(embedded_cats, nums_stacked),
        baselines=(torch.zeros_like(embedded_cats), torch.zeros_like(nums_stacked)),
        target=int(target_class),
        n_steps=n_steps,
    )
    # Split the combined categorical attribution back to per-feature
    dims = [e.embedding_dim for e in model.embeddings]
    cat_attrs: List[torch.Tensor] = []
    offset = 0
    for d in dims:
        cat_attrs.append(cat_attr[0, :, offset:offset + d].detach().cpu())  # [T, d]
        offset += d
    num_attrs: List[torch.Tensor] = [num_attr[0, :, i].detach().cpu() for i in range(num_attr.shape[-1])]
    return cat_attrs, num_attrs


def compute_shapley_values(model,
                           cats_window: Sequence[torch.Tensor],
                           nums_window: Sequence[torch.Tensor],
                           target_class: int,
                           n_samples: int = 25) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """ShapleyValueSampling via Captum. Returns the same shapes as
    `compute_integrated_gradients`. Uses zero baseline."""
    from captum.attr import ShapleyValueSampling  # type: ignore

    model.eval()
    embedded_per_cat = [emb(cats_window[j]) for j, emb in enumerate(model.embeddings)]
    embedded_cats = torch.cat(embedded_per_cat, dim=-1)
    nums_stacked = (torch.stack([n.float() for n in nums_window], dim=-1)
                    if len(nums_window) else torch.zeros(embedded_cats.shape[0], embedded_cats.shape[1], 0))
    if nums_stacked.dim() == 2:
        nums_stacked = nums_stacked.unsqueeze(0)

    def forward_for_shap(emb_cats, nums):
        return _forward_from_embeddings(model, emb_cats, nums)

    svs = ShapleyValueSampling(forward_for_shap)
    cat_attr, num_attr = svs.attribute(
        inputs=(embedded_cats, nums_stacked),
        baselines=(torch.zeros_like(embedded_cats), torch.zeros_like(nums_stacked)),
        target=int(target_class),
        n_samples=n_samples,
    )
    dims = [e.embedding_dim for e in model.embeddings]
    cat_attrs: List[torch.Tensor] = []
    offset = 0
    for d in dims:
        cat_attrs.append(cat_attr[0, :, offset:offset + d].detach().cpu())
        offset += d
    num_attrs: List[torch.Tensor] = [num_attr[0, :, i].detach().cpu() for i in range(num_attr.shape[-1])]
    return cat_attrs, num_attrs


def attributions_to_matrix(cat_attrs: Sequence[torch.Tensor],
                           num_attrs: Sequence[torch.Tensor],
                           feature_names: Sequence[str]) -> torch.Tensor:
    """Reduce per-feature-per-position attributions to a [n_features, T] matrix
    by summing the embedding dimension (for cats) and using raw values for nums."""
    rows = []
    for a in cat_attrs:
        rows.append(a.sum(dim=-1))  # [T]
    for a in num_attrs:
        rows.append(a)  # [T]
    mat = torch.stack(rows, dim=0)  # [n_features, T]
    assert mat.shape[0] == len(feature_names), \
        f"feature_names length {len(feature_names)} doesn't match rows {mat.shape[0]}"
    return mat
