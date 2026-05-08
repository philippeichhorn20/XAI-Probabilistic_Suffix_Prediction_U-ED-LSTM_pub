"""Camargo predictor adapter for the pathway explorer.

Mirrors the surface area of `RevisedPlusModelPredictor.predict_batch` so the
existing DFGAnalyzer.run_sweep code path works unchanged. Differences from the
U-ED-LSTM predictor:

- Camargo models take a flat `(cats_list, nums_list)` argument and return logits
  directly, not a `(suffix_dict, …)` tuple.
- They consume only a subset of dataset columns (cat_indices / num_indices on
  the ModelConfig) and a fixed-length window (ngram_size). This adapter projects
  + right-aligns the window before forwarding.
"""
from __future__ import annotations

from typing import List, Tuple, Sequence

import numpy as np
import torch


class CamargoModelPredictor:
    """Thin wrapper that exposes ``predict_batch`` with the same signature as
    ``RevisedPlusModelPredictor`` but forwards through a Camargo model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cat_indices: Sequence[int],
        num_indices: Sequence[int],
        ngram_size: int = 0,
    ):
        self.model = model
        self.cat_indices = tuple(cat_indices)
        self.num_indices = tuple(num_indices)
        self.ngram_size = int(ngram_size)
        self.device = next(model.parameters()).device
        self.model.eval()

    def _project_and_window(
        self,
        cat_batch: List[torch.Tensor],
        num_stacked: torch.Tensor | None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Project to the model's feature subset and right-align last `ngram_size`.

        Inputs match RevisedPlusModelPredictor.predict_batch:
        - cat_batch: List[Tensor[N, seq_len]] (one per dataset cat feature)
        - num_stacked: Tensor[N, seq_len, n_num] (or None)
        """
        cats_proj = [cat_batch[i] for i in self.cat_indices if i < len(cat_batch)]
        if num_stacked is not None and num_stacked.dim() == 3 and num_stacked.shape[2] > 0:
            n_num_total = num_stacked.shape[2]
            nums_proj = [num_stacked[:, :, i] for i in self.num_indices if i < n_num_total]
        else:
            nums_proj = []

        # Right-align last `ngram_size` events when n-gram windowing was used at training.
        if self.ngram_size > 0 and cats_proj and cats_proj[0].shape[1] > self.ngram_size:
            w = self.ngram_size
            cats_proj = [c[:, -w:] for c in cats_proj]
            nums_proj = [n[:, -w:] for n in nums_proj]
        return cats_proj, nums_proj

    @torch.no_grad()
    def predict_batch(
        self,
        cat_batch: List[torch.Tensor],
        num_stacked: torch.Tensor,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cats_proj, nums_proj = self._project_and_window(cat_batch, num_stacked)
        if not cats_proj:
            raise ValueError("CamargoModelPredictor: no categorical features after projection")

        N = cats_proj[0].shape[0]
        all_preds, all_probs = [], []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b_cat = [c[start:end].to(self.device) for c in cats_proj]
            b_num = [n[start:end].to(self.device) for n in nums_proj]
            logits = self.model((b_cat, b_num))
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
        return torch.cat(all_preds).numpy(), torch.cat(all_probs).numpy()
