"""
DFG extraction and feature-sweep analysis for the Pathway Explorer.

Pure-Python module with no Streamlit dependency. Mirrors the logic of
notebooks/helpdesk/21_pathway_explorer.ipynb and notebooks/bpic17/21_pathway_explorer.ipynb.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from src.interpretability.utils.tensor_decoder import TensorDecoder
from src.interpretability.perturbation_methods.revised_plus.revised_plus import RevisedPlusModelPredictor


# ============================================================
# Data Classes
# ============================================================

@dataclass
class DFGResult:
    """Result of DFG extraction from dataset."""
    dfg: Dict[Tuple[str, str], int]                          # (src, tgt) -> count
    successors_map: Dict[str, Dict[str, int]]                # activity -> {successor -> count}
    decision_points: Dict[str, Dict[str, int]]               # activities with >1 successor
    pathway_index: Dict[Tuple[str, ...], List[Tuple[int, int]]]  # pathway -> [(ds_idx, trace_len)]
    activity_names: List[str]                                 # idx -> name
    eos_idx: int
    seq_len: int
    n_cat: int
    n_num: int


@dataclass
class SweepFeature:
    """A single sweepable categorical feature."""
    feature_idx: int
    feature_name: str
    is_case_level: bool
    valid_values: List[int]  # encoded indices, excluding padding (0)


@dataclass
class SweepConfig:
    """Configuration for feature sweep."""
    features: List[SweepFeature]


@dataclass
class PathwayContext:
    """Context for a selected pathway: prefix tensors + baseline prediction."""
    pathway: Tuple[str, ...]
    case_index: int
    ds_idx: int
    case_id: str
    cat_tuple: tuple         # tuple of 1-D tensors (seq_len,)
    num_tuple: tuple         # tuple of 1-D tensors (seq_len,)
    prefix_len: int
    pad_len: int
    original_pred_idx: int
    original_pred_name: str
    original_pred_prob: float
    original_probs: np.ndarray  # full probability vector
    prefix_df: pd.DataFrame     # decoded prefix for display
    suffix_df: pd.DataFrame     # decoded actual suffix for display
    successors: Dict[str, int]  # observed successors at last pathway activity
    is_decision: bool


@dataclass
class SweepResult:
    """Result of a feature sweep."""
    df_sweep: pd.DataFrame
    original_pred_idx: int
    original_pred_name: str
    original_probs: np.ndarray
    branches: List[str]
    branch_indices: List[int]
    successors: Dict[str, int]
    elapsed: float


# ============================================================
# DFGAnalyzer
# ============================================================

class DFGAnalyzer:
    """Extracts DFG from dataset and runs feature sweeps."""

    def __init__(self, dataset, activity_feature: str, decoder: TensorDecoder,
                 csv_path: Optional[str] = None, csv_case_col: str = "Case ID"):
        self.dataset = dataset
        self.activity_feature = activity_feature
        self.decoder = decoder

        # Resolve activity index in cat_features
        self.act_idx = decoder.cat_features.index(activity_feature)

        # Build activity name list from decoder
        activity_idx_to_label = decoder.idx_to_label[activity_feature]
        max_idx = max(activity_idx_to_label.keys())
        self.activity_names = [
            activity_idx_to_label.get(i, f'<unk_{i}>')
            for i in range(max_idx + 1)
        ]
        self.eos_idx = next(
            i for i, name in enumerate(self.activity_names) if name == 'EOS'
        )

        # Dataset dimensions
        sample = dataset[0]
        self.n_cat = len(sample[0])
        self.n_num = len(sample[1])
        self.seq_len = sample[0][0].shape[0]

        # CSV lookup for recovering unseen encoded values (index 0 with no
        # mapping in the encoder).  Keyed by (case_id, csv_column) -> value.
        self._csv_lookup = {}
        self._csv_col_map = {}  # encoded feature name -> CSV column name
        if csv_path is not None:
            self._build_csv_lookup(csv_path, csv_case_col)

    def _build_csv_lookup(self, csv_path: str, case_col: str):
        """Load the source CSV and build a lookup for features whose encoder
        has no mapping for index 0 (i.e. unseen / out-of-vocabulary values)."""
        source_df = pd.read_csv(csv_path)
        if case_col not in source_df.columns:
            return

        # Determine which encoded features have unmapped 0 (unseen values)
        decoder = self.decoder
        unseen_features = []
        for feat_name in decoder.cat_features:
            mapping = decoder.label_to_idx[feat_name]
            if not any(v == 0 for v in mapping.values()):
                # 0 is not a real encoded value → could be unseen
                unseen_features.append(feat_name)

        # Match encoded feature names to CSV columns (exact or close match)
        csv_cols = set(source_df.columns)
        for feat_name in unseen_features:
            if feat_name in csv_cols:
                self._csv_col_map[feat_name] = feat_name

        if not self._csv_col_map:
            return

        # Build per-case lookup: take first row per case for matched columns
        cols_needed = [case_col] + list(self._csv_col_map.values())
        subset = source_df[cols_needed].drop_duplicates(subset=[case_col])
        for _, row in subset.iterrows():
            case_id = str(row[case_col])
            for feat_name, csv_col in self._csv_col_map.items():
                val = row[csv_col]
                self._csv_lookup[(case_id, feat_name)] = str(val) if pd.notna(val) else "<missing>"

    def _patch_unseen_values(self, df: pd.DataFrame, case_id: str) -> pd.DataFrame:
        """Replace '<none>' entries in a decoded DataFrame with real values
        from the source CSV lookup."""
        if not self._csv_lookup:
            return df
        for feat_name in self._csv_col_map:
            if feat_name in df.columns:
                real_val = self._csv_lookup.get((str(case_id), feat_name))
                if real_val is not None:
                    df[feat_name] = df[feat_name].replace('<none>', real_val)
        return df

    def build_dfg(self) -> DFGResult:
        """Single-pass dataset scan to extract DFG, pathways, and decision points."""
        dataset = self.dataset
        seq_len = self.seq_len
        act_idx = self.act_idx
        eos_idx = self.eos_idx
        activity_names = self.activity_names

        # Collect unique cases (longest trace per case_id)
        cases = {}
        for i in range(len(dataset)):
            cat_t, num_t, case_id = dataset[i]
            trace_len = int((cat_t[0] != 0).sum().item())
            if case_id not in cases or trace_len > cases[case_id][1]:
                cases[case_id] = (i, trace_len)

        # Build DFG and pathway index
        dfg = {}
        pathway_index = {}

        for case_id, (ds_idx, trace_len) in cases.items():
            cat_t = dataset[ds_idx][0]
            src_start = seq_len - trace_len
            acts = cat_t[act_idx][src_start:src_start + trace_len]

            # Strip trailing EOS
            useful_len = trace_len
            for j in range(trace_len - 1, -1, -1):
                if acts[j].item() == eos_idx:
                    useful_len = j
                else:
                    break

            act_names = [activity_names[acts[j].item()] for j in range(useful_len)]

            # Record DFG edges
            for k in range(len(act_names) - 1):
                edge = (act_names[k], act_names[k + 1])
                dfg[edge] = dfg.get(edge, 0) + 1

            # Record all prefix pathways
            for k in range(1, len(act_names) + 1):
                pathway = tuple(act_names[:k])
                pathway_index.setdefault(pathway, []).append((ds_idx, trace_len))

        # Build successor map and decision points
        successors_map = {}
        for (src, tgt), count in dfg.items():
            successors_map.setdefault(src, {})[tgt] = count

        decision_points = {
            act: succs for act, succs in successors_map.items()
            if len(succs) > 1
        }

        return DFGResult(
            dfg=dfg,
            successors_map=successors_map,
            decision_points=decision_points,
            pathway_index=pathway_index,
            activity_names=self.activity_names,
            eos_idx=self.eos_idx,
            seq_len=self.seq_len,
            n_cat=self.n_cat,
            n_num=self.n_num,
        )

    def get_sweepable_features(self, case_level_cat_names: Optional[List[str]]) -> SweepConfig:
        """Determine which categorical features to sweep.

        All non-activity categorical features are sweepable.
        Case-level features are matched by name against decoder.cat_features,
        silently ignoring names that don't match (handles mismatches like
        'VariantIndex' vs 'Variant index').
        """
        decoder = self.decoder
        vocab_sizes = decoder.get_vocab_sizes()
        cat_features = decoder.cat_features

        # Resolve case-level names to feature indices
        case_level_indices = set()
        if case_level_cat_names:
            for name in case_level_cat_names:
                if name in cat_features:
                    case_level_indices.add(cat_features.index(name))

        features = []
        for ci, feat_name in enumerate(cat_features):
            if ci == self.act_idx:
                continue  # never sweep the activity column
            valid_vals = list(range(1, vocab_sizes[ci]))  # skip padding (0)
            if not valid_vals:
                continue
            features.append(SweepFeature(
                feature_idx=ci,
                feature_name=feat_name,
                is_case_level=ci in case_level_indices,
                valid_values=valid_vals,
            ))

        return SweepConfig(features=features)

    def build_pathway_context(
        self,
        dfg_result: DFGResult,
        pathway: Tuple[str, ...],
        case_index: int,
        model: torch.nn.Module,
    ) -> PathwayContext:
        """Build prefix tensor for a pathway + case, run baseline prediction."""
        dataset = self.dataset
        seq_len = self.seq_len
        activity_names = self.activity_names

        matching_cases = dfg_result.pathway_index[pathway]
        ds_idx, trace_len = matching_cases[case_index]
        cat_full, num_full, case_id = dataset[ds_idx]

        prefix_len = len(pathway)
        pad_len = seq_len - prefix_len
        src_start = seq_len - trace_len

        # Build padded prefix tensors
        cat_tuple = []
        for c in cat_full:
            t = torch.zeros_like(c)
            t[pad_len:] = c[src_start:src_start + prefix_len]
            cat_tuple.append(t)
        cat_tuple = tuple(cat_tuple)

        num_tuple = []
        for n in num_full:
            t = torch.zeros_like(n)
            t[pad_len:] = n[src_start:src_start + prefix_len]
            num_tuple.append(t)
        num_tuple = tuple(num_tuple)

        # Run model prediction
        cat_in = [c.unsqueeze(0) for c in cat_tuple]
        num_in = [n.unsqueeze(0) for n in num_tuple]
        with torch.no_grad():
            preds = model((cat_in, num_in))[0]
            logits = preds[0][f'{self.activity_feature}_mean'][0]
            p = torch.softmax(logits, dim=-1).squeeze(0)
            top_p, top_idx = p.max(dim=-1)

        original_pred_idx = top_idx.item()
        original_pred_name = activity_names[original_pred_idx]
        original_pred_prob = top_p.item()
        original_probs = p.numpy()

        # Decode prefix and suffix from the original full trace (avoids
        # showing "<pad>" for non-activity features that legitimately have
        # encoded index 0 in the re-padded tensors).
        prefix_cat = tuple(c[src_start:src_start + prefix_len] for c in cat_full)
        prefix_num = tuple(n[src_start:src_start + prefix_len] for n in num_full)
        prefix_df = self.decoder.decode_sequence(prefix_cat, prefix_num, case_id=case_id)
        prefix_df = self._patch_unseen_values(prefix_df, case_id)

        suffix_start = src_start + prefix_len
        suffix_cat = tuple(c[suffix_start:] for c in cat_full)
        suffix_num = tuple(n[suffix_start:] for n in num_full)
        suffix_df = self.decoder.decode_sequence(suffix_cat, suffix_num, case_id=case_id)
        suffix_df = self._patch_unseen_values(suffix_df, case_id)

        # Successors at last pathway activity
        last_act = pathway[-1]
        successors = dfg_result.successors_map.get(last_act, {})
        is_decision = len(successors) > 1

        return PathwayContext(
            pathway=pathway,
            case_index=case_index,
            ds_idx=ds_idx,
            case_id=case_id,
            cat_tuple=cat_tuple,
            num_tuple=num_tuple,
            prefix_len=prefix_len,
            pad_len=pad_len,
            original_pred_idx=original_pred_idx,
            original_pred_name=original_pred_name,
            original_pred_prob=original_pred_prob,
            original_probs=original_probs,
            prefix_df=prefix_df,
            suffix_df=suffix_df,
            successors=successors,
            is_decision=is_decision,
        )

    def run_sweep(
        self,
        context: PathwayContext,
        sweep_config: SweepConfig,
        predictor: RevisedPlusModelPredictor,
    ) -> SweepResult:
        """Run feature sweep over all sweepable features. Returns SweepResult."""
        decoder = self.decoder
        activity_names = self.activity_names
        n_cat = self.n_cat
        pad_len = context.pad_len
        prefix_len = context.prefix_len
        original_pred_idx = context.original_pred_idx

        frozen_cat = [c.clone() for c in context.cat_tuple]
        frozen_num_stacked = torch.stack(list(context.num_tuple), dim=-1)

        sweep_results = []
        t0 = time.time()

        for sf in sweep_config.features:
            ci = sf.feature_idx
            feat_name = sf.feature_name
            is_case_level = sf.is_case_level
            vals = sf.valid_values
            N = len(vals)

            positions = [0] if is_case_level else list(range(prefix_len))

            for pos_idx in positions:
                pos = pad_len + pos_idx

                # Build batch: N variants, one per valid value
                cat_batch = []
                for cj in range(n_cat):
                    if cj == ci:
                        t = frozen_cat[cj].unsqueeze(0).expand(N, -1).clone()
                        for k, val in enumerate(vals):
                            if is_case_level:
                                t[k, pad_len:] = val
                            else:
                                t[k, pos] = val
                        cat_batch.append(t)
                    else:
                        cat_batch.append(frozen_cat[cj].unsqueeze(0).expand(N, -1).clone())

                num_stacked = frozen_num_stacked.unsqueeze(0).expand(N, -1, -1).clone()

                preds_batch, probs_batch = predictor.predict_batch(cat_batch, num_stacked)

                original_val = frozen_cat[ci][pos].item()
                for k, val in enumerate(vals):
                    pred_class = int(preds_batch[k])
                    sweep_results.append({
                        'feature': feat_name,
                        'feature_idx': ci,
                        'position': pos_idx + 1,
                        'is_case_level': is_case_level,
                        'value_idx': val,
                        'value_name': decoder.decode_categorical_value(ci, val),
                        'is_original': val == original_val,
                        'predicted_class': pred_class,
                        'predicted_class_name': activity_names[pred_class],
                        'prob_original_class': float(probs_batch[k, original_pred_idx]),
                        'prob_predicted_class': float(probs_batch[k, pred_class]),
                        'probability_vector': probs_batch[k].tolist(),
                    })

        elapsed = time.time() - t0
        df_sweep = pd.DataFrame(sweep_results)

        branches = list(context.successors.keys())
        branch_indices = [activity_names.index(b) for b in branches]

        return SweepResult(
            df_sweep=df_sweep,
            original_pred_idx=context.original_pred_idx,
            original_pred_name=context.original_pred_name,
            original_probs=context.original_probs,
            branches=branches,
            branch_indices=branch_indices,
            successors=context.successors,
            elapsed=elapsed,
        )
