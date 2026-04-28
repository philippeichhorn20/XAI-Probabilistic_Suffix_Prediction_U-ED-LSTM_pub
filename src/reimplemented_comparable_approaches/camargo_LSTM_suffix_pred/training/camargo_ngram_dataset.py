"""N-gram dataset wrapper for the Camargo reimplementation.

The shared U-ED-LSTM `EventLogDataset` produces fixed-size windows padded with
EOS on the right so the main project's encoder-decoder can learn to terminate.
Reusing those windows as "predict the last column given the rest" trains the
Camargo single-step model to emit EOS ~90% of the time, which wrecks accuracy
at short prefixes (see Camargo et al., BPM 2019, §3.1 "Sequences creation").

This wrapper reads the existing encoded pickle (no CSV re-encoding), recovers
the original trace per case, and emits one sample per time-step following the
paper's n-gram convention:

    step k: prefix = right-aligned last `ngram_size` events up to position k,
            pad-left with zeros; target = activity at position k+1.

The U-ED-LSTM loader is not modified — this class is only consumed by the
Camargo training/inference notebooks.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class CamargoNGramDataset(Dataset):
    def __init__(self,
                 base_dataset,
                 ngram_size: int,
                 activity_idx: int = 0,
                 eos_idx: int = 5,
                 cat_indices=None,
                 num_indices=None,
                 include_step_zero: bool = True):
        """
        Args:
            base_dataset: an `EventLogDataset` — each sample is
                (cats: list[[window]], nums: list[[window]], case_id).
            ngram_size: fixed-width sliding window. Paper uses 5 for Helpdesk, 15 for BPI 2012.
            activity_idx: position of the activity feature inside the base cats list
                (BEFORE projection). Used both to locate EOS and to pick the target.
            eos_idx: integer index of the EOS activity (trace terminator).
            cat_indices: optional iterable selecting a subset of cat features to
                keep in each sample (e.g. (0, 2) to drop everything but
                concept:name + org:resource). `None` keeps all. Must include
                `activity_idx` if specified.
            num_indices: same idea for numerical features. `None` keeps all.
            include_step_zero: if True, emit a "predict first activity from an
                all-zero window" sample per case (matches paper's Table 1).
        """
        self.ngram_size = ngram_size
        self.all_categories = base_dataset.all_categories
        self.cat_indices = tuple(cat_indices) if cat_indices is not None else None
        self.num_indices = tuple(num_indices) if num_indices is not None else None

        # Target activity idx AFTER projection — where does activity_idx land
        # in the projected cats list?
        if self.cat_indices is not None:
            assert activity_idx in self.cat_indices, (
                f"activity_idx={activity_idx} must appear in cat_indices={self.cat_indices}"
            )
            self._projected_activity_idx = self.cat_indices.index(activity_idx)
        else:
            self._projected_activity_idx = activity_idx

        cases = self._recover_cases(
            base_dataset, activity_idx, eos_idx, self.cat_indices, self.num_indices
        )
        self.samples: List[Tuple[List[torch.Tensor], List[torch.Tensor], int]] = []
        for case_cats, case_nums in cases:
            self._emit_samples_for_case(
                case_cats, case_nums, self._projected_activity_idx, include_step_zero
            )

    @staticmethod
    def _recover_cases(base_dataset, activity_idx: int, eos_idx: int,
                       cat_indices, num_indices):
        """Single pass over the base dataset. For each case_id, remember the
        first window that actually contains EOS — the base dataset emits
        multiple progressively-right-shifted windows per case, and the earliest
        one often stops mid-trace. Once a case's EOS-containing window is
        captured, subsequent windows for the same case are skipped.

        `cat_indices` / `num_indices` project each recovered case down to the
        feature subset the downstream model actually consumes.
        """
        found: dict = {}  # cid -> (cats_raw, nums_raw, acts_list)
        for i in range(len(base_dataset)):
            cats_raw, nums_raw, cid = base_dataset[i]
            if cid in found:
                continue
            acts = cats_raw[activity_idx].squeeze().tolist()
            if eos_idx in acts:
                found[cid] = (cats_raw, nums_raw, acts)

        cases = []
        for cats_raw, nums_raw, acts in found.values():
            start = next((k for k, a in enumerate(acts) if a != 0), None)
            if start is None:
                continue
            end = None
            for k in range(start, len(acts)):
                if acts[k] == eos_idx:
                    end = k
                    break
            if end is None:
                continue

            trace_len = end - start + 1  # includes EOS
            case_cats = [c.squeeze()[start:start + trace_len].clone() for c in cats_raw]
            case_nums = [n.squeeze()[start:start + trace_len].clone() for n in nums_raw]
            if cat_indices is not None:
                case_cats = [case_cats[i] for i in cat_indices]
            if num_indices is not None:
                case_nums = [case_nums[i] for i in num_indices]
            cases.append((case_cats, case_nums))
        return cases

    def _emit_samples_for_case(self,
                               case_cats: List[torch.Tensor],
                               case_nums: List[torch.Tensor],
                               activity_idx: int,
                               include_step_zero: bool):
        """One sample per time-step k; target = event at position k+1."""
        L = case_cats[0].shape[0]  # trace length including EOS
        N = self.ngram_size

        if include_step_zero and L > 0:
            target0 = int(case_cats[activity_idx][0].item())
            cats_win = [torch.zeros(N, dtype=c.dtype) for c in case_cats]
            nums_win = [torch.zeros(N, dtype=n.dtype) for n in case_nums]
            self.samples.append((cats_win, nums_win, target0))

        for k in range(L - 1):
            target = int(case_cats[activity_idx][k + 1].item())
            history = min(k + 1, N)
            cats_win: List[torch.Tensor] = []
            for c in case_cats:
                w = torch.zeros(N, dtype=c.dtype)
                w[N - history:] = c[k + 1 - history:k + 1]
                cats_win.append(w)
            nums_win: List[torch.Tensor] = []
            for n in case_nums:
                w = torch.zeros(N, dtype=n.dtype)
                w[N - history:] = n[k + 1 - history:k + 1]
                nums_win.append(w)
            self.samples.append((cats_win, nums_win, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
