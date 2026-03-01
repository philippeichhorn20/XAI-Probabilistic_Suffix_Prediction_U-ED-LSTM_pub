"""
REVISED+: Counterfactual explanations with VAE feasibility and Declare plausibility.

Based on Stevens et al. (2024) - "Generating Feasible and Plausible Counterfactual
Explanations for Outcome Prediction of Business Processes" (arXiv:2403.09232).

Adapted for multi-feature sequences and next-activity prediction via latent space
search. The original paper uses DiCE's gradient-based optimization; since our data
is discrete sequences and the prediction model is not differentiable through the VAE,
we use iterative elite-sampling in the VAE latent space instead.

Algorithm:
    1. Train LSTM-VAE on training traces (feasibility model)
    2. Mine Declare TLC constraints from all traces (plausibility model)
    3. For each explanation request:
       a. Encode original prefix to latent z
       b. Iteratively sample z candidates around z, decode, predict
       c. Score valid counterfactuals (changed prediction) with:
          - Proximity (distance to original)
          - Sparsity (number of changes)
          - Feasibility (VAE reconstruction loss)
          - Plausibility (Declare constraint satisfaction)
       d. Return diverse top-k counterfactuals

Usage:
    >>> from src.interpretability.perturbation_methods import (
    ...     RevisedPlus, RevisedPlusConfig, create_revised_plus_for_model
    ... )
    >>> config = RevisedPlusConfig(vae_epochs=100, device='mps')
    >>> rp = create_revised_plus_for_model(model, train_dataset, activity_names, config)
    >>> explanation = rp.explain(cat_tensors, num_tensors, target_class=5)
    >>> print(explanation)
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from torch.utils.data import DataLoader

from .simple_vae import SimpleSequenceVAE, train_vae
from ..declare import DeclareConstraint, DeclareConstraintChecker, DeclareConstraintMiner


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RevisedPlusConfig:
    """Configuration for REVISED+ counterfactual generation."""

    # VAE architecture
    vae_embed_dim: int = 24
    vae_hidden_dim: int = 96
    vae_latent_dim: int = 24
    vae_num_layers: int = 1
    vae_dropout: float = 0.0

    # VAE training
    vae_epochs: int = 100
    vae_lr: float = 1e-3
    vae_kl_weight: float = 0.1
    vae_batch_size: int = 64
    vae_grad_clip: float = 1.0
    vae_constraint_weight: float = 1.0

    # Declare constraint mining (TLC only — no DLC for next-activity prediction)
    declare_min_support: float = 0.9
    declare_templates: Optional[List] = None

    # Latent space search
    n_candidates_per_round: int = 200
    n_search_rounds: int = 5
    initial_noise_scale: float = 1.0
    noise_decay: float = 0.7
    min_noise_scale: float = 0.1
    elite_fraction: float = 0.1

    # Scoring weights (REVISED+ objective)
    w_proximity: float = 1.0
    w_sparsity: float = 1.0
    w_feasibility: float = 1.0
    w_plausibility: float = 1.0

    # Plausibility filtering
    min_plausibility: float = 0.0  # minimum plausibility_optimistic to accept a counterfactual

    # Length constraint: max allowed deviation from original prefix length.
    # None = no constraint (any length). 0 = exact same length.
    # 2 = allow ±2 events difference.
    max_length_delta: Optional[int] = 2

    # Output
    top_k: int = 5
    diversity_threshold: float = 0.3

    # General
    suffix_step: int = 0
    seed: int = 42
    verbose: bool = True
    device: str = "cpu"


# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class RevisedPlusCounterfactual:
    """A single counterfactual instance with quality metrics."""

    cat_sequence: List[torch.Tensor]
    num_sequence: Optional[torch.Tensor]
    activity_sequence: List[int]

    counterfactual_prediction: int
    counterfactual_prediction_name: str
    counterfactual_probability: float

    proximity: float
    sparsity: int
    feasibility: float
    plausibility_definite: float   # prefix-safe constraints only (used in penalty)
    plausibility_optimistic: float  # all constraints, prefix as complete (validity gate)
    combined_score: float

    z: Optional[torch.Tensor] = None


@dataclass
class RevisedPlusExplanation:
    """Full REVISED+ explanation result."""

    original_cat: List[torch.Tensor]
    original_num: Optional[torch.Tensor]
    original_activity_sequence: List[int]
    original_prediction: int
    original_prediction_name: str
    original_probability: float

    target_class: Optional[int]
    target_class_name: Optional[str]

    counterfactuals: List[RevisedPlusCounterfactual]

    n_candidates_evaluated: int
    n_valid_counterfactuals: int
    search_time_seconds: float

    n_prefix_safe_constraints: int
    n_all_constraints: int

    prefix_len: int = 0

    def __str__(self) -> str:
        lines = [
            "REVISED+ Counterfactual Explanation",
            "=" * 60,
            f"Original: {self.original_prediction_name} "
            f"(idx={self.original_prediction}, p={self.original_probability:.3f})",
            f"Prefix length: {self.prefix_len} events",
            f"Target: {self.target_class_name or 'any different class'}",
            f"Constraints: {self.n_prefix_safe_constraints} prefix-safe, "
            f"{self.n_all_constraints} total",
            f"Search: {self.n_candidates_evaluated} evaluated, "
            f"{self.n_valid_counterfactuals} valid, "
            f"{self.search_time_seconds:.1f}s",
            "",
        ]
        if not self.counterfactuals:
            lines.append("No counterfactuals found.")
        else:
            lines.append(f"Top {len(self.counterfactuals)} counterfactuals:")
            for i, cf in enumerate(self.counterfactuals):
                lines.append(
                    f"  [{i+1}] -> {cf.counterfactual_prediction_name} "
                    f"(p={cf.counterfactual_probability:.3f}) | "
                    f"prox={cf.proximity:.3f} sparse={cf.sparsity} "
                    f"feas={cf.feasibility:.3f} "
                    f"plaus_def={cf.plausibility_definite:.2f} "
                    f"plaus_opt={cf.plausibility_optimistic:.2f} "
                    f"score={cf.combined_score:.4f}"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prediction": self.original_prediction,
            "original_prediction_name": self.original_prediction_name,
            "original_probability": self.original_probability,
            "target_class": self.target_class,
            "prefix_len": self.prefix_len,
            "n_counterfactuals": len(self.counterfactuals),
            "n_candidates_evaluated": self.n_candidates_evaluated,
            "n_valid_counterfactuals": self.n_valid_counterfactuals,
            "search_time_seconds": self.search_time_seconds,
            "n_prefix_safe_constraints": self.n_prefix_safe_constraints,
            "n_all_constraints": self.n_all_constraints,
            "counterfactuals": [
                {
                    "prediction": cf.counterfactual_prediction,
                    "prediction_name": cf.counterfactual_prediction_name,
                    "probability": cf.counterfactual_probability,
                    "proximity": cf.proximity,
                    "sparsity": cf.sparsity,
                    "feasibility": cf.feasibility,
                    "plausibility_definite": cf.plausibility_definite,
                    "plausibility_optimistic": cf.plausibility_optimistic,
                    "combined_score": cf.combined_score,
                    "activity_sequence": cf.activity_sequence,
                }
                for cf in self.counterfactuals
            ],
        }

    def get_best(self) -> Optional[RevisedPlusCounterfactual]:
        return self.counterfactuals[0] if self.counterfactuals else None

    def get_counterfactuals_for_class(
        self, class_idx: int
    ) -> List[RevisedPlusCounterfactual]:
        return [
            cf
            for cf in self.counterfactuals
            if cf.counterfactual_prediction == class_idx
        ]


# =============================================================================
# Model Predictor
# =============================================================================

class RevisedPlusModelPredictor:
    """
    Wraps U-ED-LSTM for prediction on full (cat_tensors, num_tensors) tuples.

    The model's forward() expects:
        prefixes = [cat_tensors: List[Tensor[batch, seq]], num_tensors: List[Tensor[batch, seq]]]

    The VAE produces:
        num_out: Tensor[batch, seq, n_num]

    This class bridges the two formats.
    """

    def __init__(self, model: torch.nn.Module, suffix_step: int = 0):
        self.model = model
        self.suffix_step = suffix_step
        self.device = next(model.parameters()).device
        self.model.eval()

    def predict(
        self,
        cat_tensors: List[torch.Tensor],
        num_tensors: List[torch.Tensor],
    ) -> int:
        cat_dev = [t.to(self.device) for t in cat_tensors]
        num_dev = [t.to(self.device) for t in num_tensors]

        with torch.no_grad():
            # Model returns: predictions, (h,c), seq_len_pred, output_feature_indices
            # predictions = [cat_dict, num_dict]
            predictions = self.model((cat_dev, num_dev))[0]
            logits = predictions[0]["Activity_mean"][self.suffix_step]
            return logits.argmax(dim=-1).item()

    def predict_proba(
        self,
        cat_tensors: List[torch.Tensor],
        num_tensors: List[torch.Tensor],
    ) -> np.ndarray:
        cat_dev = [t.to(self.device) for t in cat_tensors]
        num_dev = [t.to(self.device) for t in num_tensors]

        with torch.no_grad():
            predictions = self.model((cat_dev, num_dev))[0]
            logits = predictions[0]["Activity_mean"][self.suffix_step]
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy().squeeze(0)

    def predict_batch(
        self,
        cat_batch: List[torch.Tensor],
        num_stacked: torch.Tensor,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for search efficiency.

        Args:
            cat_batch: List of [N, seq_len] tensors (one per cat feature).
            num_stacked: [N, seq_len, n_num] tensor (VAE format).

        Returns:
            (predictions [N], probabilities [N, n_classes])
        """
        N = cat_batch[0].shape[0]
        n_num = num_stacked.shape[2] if num_stacked is not None else 0
        all_preds = []
        all_probs = []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b_cat = [c[start:end].to(self.device) for c in cat_batch]
            # Convert stacked num [batch, seq, n_num] -> List[Tensor[batch, seq]]
            if n_num > 0:
                b_num = [
                    num_stacked[start:end, :, i].to(self.device)
                    for i in range(n_num)
                ]
            else:
                b_num = []

            with torch.no_grad():
                predictions = self.model((b_cat, b_num))[0]
                logits = predictions[0]["Activity_mean"][self.suffix_step]
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

        return torch.cat(all_preds).numpy(), torch.cat(all_probs).numpy()


# =============================================================================
# Main Orchestrator
# =============================================================================

class RevisedPlus:
    """
    REVISED+: Counterfactual explanations via VAE feasibility + Declare plausibility.

    Generates counterfactual explanations by searching the VAE latent space for
    sequences that change the prediction while remaining feasible and plausible.

    Example:
        >>> config = RevisedPlusConfig(vae_epochs=100, device='mps')
        >>> rp = RevisedPlus.from_dataset(model, train_dataset, activity_names, config)
        >>> explanation = rp.explain(cat_tensors, num_tensors, target_class=5)
        >>> print(explanation)
    """

    def __init__(
        self,
        predictor: RevisedPlusModelPredictor,
        vae: Optional[SimpleSequenceVAE],
        cat_vocab_sizes: List[int],
        num_features: int,
        seq_len: int,
        activity_names: List[str],
        all_constraints: Set[DeclareConstraint],
        prefix_safe_constraints: Set[DeclareConstraint],
        config: RevisedPlusConfig,
    ):
        self.predictor = predictor
        self.vae = vae
        self.cat_vocab_sizes = cat_vocab_sizes
        self.num_features = num_features
        self.seq_len = seq_len
        self.activity_names = activity_names
        self.all_constraints = all_constraints
        self.prefix_safe_constraints = prefix_safe_constraints
        self.config = config
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    @classmethod
    def from_dataset(
        cls,
        model: torch.nn.Module,
        dataset,
        activity_names: List[str],
        config: Optional[RevisedPlusConfig] = None,
        pretrained_vae: Optional[SimpleSequenceVAE] = None,
        vae_path: Optional[str] = None,
    ) -> "RevisedPlus":
        """Factory: build from dataset. Trains VAE, mines constraints.

        Args:
            vae_path: Path to save/load the VAE model. If the file exists,
                     the VAE is loaded instead of trained. After training,
                     the VAE is saved to this path.
        """
        config = config or RevisedPlusConfig()
        predictor = RevisedPlusModelPredictor(model, suffix_step=config.suffix_step)

        n_cat = len(dataset[0][0])
        n_num = len(dataset[0][1])
        seq_len = dataset[0][0][0].shape[0]

        cat_max = [0] * n_cat
        for item in dataset:
            for i, cat_tensor in enumerate(item[0]):
                cat_max[i] = max(cat_max[i], int(cat_tensor.max().item()))
        cat_vocab_sizes = [m + 1 for m in cat_max]

        rp = cls(
            predictor=predictor,
            vae=None,
            cat_vocab_sizes=cat_vocab_sizes,
            num_features=n_num,
            seq_len=seq_len,
            activity_names=activity_names,
            all_constraints=set(),
            prefix_safe_constraints=set(),
            config=config,
        )
        rp.fit(dataset, pretrained_vae=pretrained_vae, vae_path=vae_path)
        return rp

    # =========================================================================
    # Fitting
    # =========================================================================

    def fit(
        self,
        dataset,
        pretrained_vae: Optional[SimpleSequenceVAE] = None,
        vae_path: Optional[str] = None,
    ) -> "RevisedPlus":
        """
        Train VAE on random-length prefixes and mine Declare constraints.

        The VAE is trained on randomly truncated prefixes (not full traces)
        so it learns the prefix manifold. This means decoded outputs are
        naturally prefix-shaped and can vary in length.

        Args:
            dataset: EventLogDataset — items are (cat_tuple, num_tuple, case_id).
            pretrained_vae: Skip VAE training if provided.
            vae_path: Path to save/load the VAE model. If the file exists,
                     the VAE is loaded instead of trained. After training,
                     the VAE is saved to this path.
        """
        n_cat = len(dataset[0][0])
        seq_len = self.seq_len

        # --- DataLoader with prefix augmentation (left-padded) ---
        # Data is LEFT-PADDED: padding zeros at the start, events at the end.
        # For each trace, sample a prefix length k ~ Uniform(1, trace_length),
        # take the first k events, and left-pad back to seq_len.
        def collate_fn(batch):
            cat_batch_list = [[] for _ in range(n_cat)]
            num_batch_list = []

            for item in batch:
                cat_tuple, num_tuple, _ = item

                # Detect actual trace length and event start position
                trace_len = int((cat_tuple[0] != 0).sum().item())
                event_start = seq_len - trace_len
                # Random prefix length: at least 1, at most full trace
                k = torch.randint(1, max(2, trace_len + 1), (1,)).item()

                # Extract first k events and left-pad to seq_len
                new_start = seq_len - k
                for i in range(n_cat):
                    t = torch.zeros_like(cat_tuple[i])
                    t[new_start:] = cat_tuple[i][event_start:event_start + k]
                    cat_batch_list[i].append(t)

                num_stacked = torch.stack(num_tuple, dim=-1)  # [seq_len, n_num]
                num_new = torch.zeros_like(num_stacked)
                num_new[new_start:] = num_stacked[event_start:event_start + k]
                num_batch_list.append(num_new)

            cat_batch = [torch.stack(cat_batch_list[i]) for i in range(n_cat)]
            num_batch = torch.stack(num_batch_list)
            return cat_batch, num_batch

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.vae_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        # --- VAE: load from file, accept pretrained, or train from scratch ---
        vae_loaded = False
        if pretrained_vae is not None:
            self.vae = pretrained_vae.to(self.device)
            vae_loaded = True
        elif vae_path is not None and Path(vae_path).exists():
            if self.config.verbose:
                print(f"Loading VAE from {vae_path}")
            self.vae = torch.load(vae_path, weights_only=False)
            self.vae = self.vae.to(self.device)
            vae_loaded = True
        else:
            self.vae = SimpleSequenceVAE(
                cat_vocab_sizes=self.cat_vocab_sizes,
                num_features=self.num_features,
                seq_len=self.seq_len,
                embed_dim=self.config.vae_embed_dim,
                hidden_dim=self.config.vae_hidden_dim,
                latent_dim=self.config.vae_latent_dim,
                num_layers=self.config.vae_num_layers,
                dropout=self.config.vae_dropout,
            )
            train_vae(
                self.vae,
                dataloader,
                epochs=self.config.vae_epochs,
                lr=self.config.vae_lr,
                kl_weight=self.config.vae_kl_weight,
                device=self.config.device,
                verbose=self.config.verbose,
                grad_clip=self.config.vae_grad_clip,
            )
            self.vae = self.vae.to(self.device)
            # Save trained VAE for reuse
            if vae_path is not None:
                Path(vae_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.vae.cpu(), vae_path)
                self.vae = self.vae.to(self.device)
                if self.config.verbose:
                    print(f"VAE saved to {vae_path}")
        self.vae.eval()

        # --- Extract activity sequences (full traces for constraint mining) ---
        if self.config.verbose:
            print("Extracting activity sequences...")
        activity_sequences = []
        for i in range(len(dataset)):
            cat_tuple, _, _ = dataset[i]
            act_seq = [x.item() for x in cat_tuple[0] if x.item() != 0]
            activity_sequences.append(act_seq)

        # --- Mine Declare constraints ---
        activity_vocab_size = self.cat_vocab_sizes[0]
        miner = DeclareConstraintMiner(
            vocab_size=activity_vocab_size,
            min_support=self.config.declare_min_support,
        )
        self.all_constraints = miner.mine(
            activity_sequences, templates=self.config.declare_templates
        )
        self.prefix_safe_constraints = set(
            DeclareConstraintChecker.prefix_safe_constraints(self.all_constraints)
        )
        if self.config.verbose:
            print(
                f"Mined {len(self.all_constraints)} constraints "
                f"({len(self.prefix_safe_constraints)} prefix-safe)"
            )

        return self

    # =========================================================================
    # Explanation
    # =========================================================================

    def explain(
        self,
        cat_tensors: List[torch.Tensor],
        num_tensors: Union[List[torch.Tensor], torch.Tensor, None] = None,
        target_class: Optional[int] = None,
        fixed_prefix_len: int = 0,
    ) -> RevisedPlusExplanation:
        """
        Generate counterfactual explanation for a prefix.

        The counterfactual is an alternative prefix (same length) that produces
        a different next-activity prediction. The VAE decodes full sequences but
        we truncate to the original prefix length and re-pad, so that prediction
        changes come from content changes, not length changes.

        Args:
            cat_tensors: List of tensors, each [1, seq_len] or [seq_len].
            num_tensors: Numerical features — either List[Tensor] (model format)
                        or Tensor[1, seq_len, n_num] (VAE format) or None.
            target_class: Target prediction class. None = any different class.
            fixed_prefix_len: Number of leading positions to keep fixed (0 = all
                             positions within the prefix can change).
        """
        start_time = time.time()

        # Normalize inputs
        cat_b, num_stacked, num_list = self._normalize_inputs(cat_tensors, num_tensors)

        # Detect actual prefix length (non-padding positions)
        actual_prefix_len = int((cat_b[0].squeeze(0) != 0).sum().item())

        # Original prediction
        original_pred, original_prob, original_probs = self._get_prediction(
            cat_b, num_list
        )
        original_act_seq = self._extract_activity_sequence(cat_b)

        if self.config.verbose:
            print(
                f"Original: {self.activity_names[original_pred] if original_pred < len(self.activity_names) else original_pred} "
                f"(p={original_prob:.3f}), prefix_len={actual_prefix_len}"
            )

        # Encode to VAE latent space
        self.vae.eval()
        with torch.no_grad():
            z_orig, _ = self.vae.encode(
                [c.to(self.device) for c in cat_b], num_stacked.to(self.device)
            )

        # Search
        valid_cfs = self._search_latent_space(
            z_orig=z_orig,
            original_cat=cat_b,
            original_num_stacked=num_stacked,
            original_num_list=num_list,
            original_pred=original_pred,
            target_class=target_class,
            fixed_prefix_len=fixed_prefix_len,
            actual_prefix_len=actual_prefix_len,
        )

        # Select diverse top-k
        diverse_cfs = self._select_diverse(valid_cfs, self.config.top_k)

        elapsed = time.time() - start_time
        target_name = (
            self.activity_names[target_class]
            if target_class is not None and target_class < len(self.activity_names)
            else None
        )

        return RevisedPlusExplanation(
            original_cat=[c.squeeze(0).cpu() for c in cat_b],
            original_num=num_stacked.squeeze(0).cpu() if num_stacked is not None else None,
            original_activity_sequence=original_act_seq,
            original_prediction=original_pred,
            original_prediction_name=self.activity_names[original_pred]
            if original_pred < len(self.activity_names)
            else str(original_pred),
            original_probability=original_prob,
            target_class=target_class,
            target_class_name=target_name,
            counterfactuals=diverse_cfs,
            n_candidates_evaluated=self.config.n_candidates_per_round
            * self.config.n_search_rounds,
            n_valid_counterfactuals=len(valid_cfs),
            search_time_seconds=elapsed,
            n_prefix_safe_constraints=len(self.prefix_safe_constraints),
            n_all_constraints=len(self.all_constraints),
            prefix_len=actual_prefix_len,
        )

    # =========================================================================
    # Input Normalization
    # =========================================================================

    def _normalize_inputs(
        self,
        cat_tensors: List[torch.Tensor],
        num_tensors: Union[List[torch.Tensor], torch.Tensor, None],
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """
        Normalize inputs to consistent formats.

        Returns:
            cat_b: List[Tensor[1, seq_len]] — batched categorical tensors.
            num_stacked: Tensor[1, seq_len, n_num] — for VAE.
            num_list: List[Tensor[1, seq_len]] — for model predictor.
        """
        # Ensure batch dim on categoricals
        cat_b = []
        for c in cat_tensors:
            if c.dim() == 1:
                cat_b.append(c.unsqueeze(0))
            else:
                cat_b.append(c)

        # Handle numerical inputs
        if num_tensors is None:
            seq_len = cat_b[0].shape[1]
            num_stacked = torch.zeros(1, seq_len, 0)
            num_list = []
        elif isinstance(num_tensors, torch.Tensor):
            # Already stacked [1, seq_len, n_num] or [seq_len, n_num]
            if num_tensors.dim() == 2:
                num_stacked = num_tensors.unsqueeze(0)
            else:
                num_stacked = num_tensors
            num_list = [num_stacked[:, :, i] for i in range(num_stacked.shape[2])]
        else:
            # List of tensors (model format)
            num_list = []
            for n in num_tensors:
                if n.dim() == 1:
                    num_list.append(n.unsqueeze(0))
                else:
                    num_list.append(n)
            if num_list:
                num_stacked = torch.stack(num_list, dim=-1)
            else:
                seq_len = cat_b[0].shape[1]
                num_stacked = torch.zeros(1, seq_len, 0)

        return cat_b, num_stacked, num_list

    # =========================================================================
    # Prediction Helpers
    # =========================================================================

    def _get_prediction(
        self,
        cat_b: List[torch.Tensor],
        num_list: List[torch.Tensor],
    ) -> Tuple[int, float, np.ndarray]:
        probs = self.predictor.predict_proba(cat_b, num_list)
        pred = int(np.argmax(probs))
        prob = float(probs[pred])
        return pred, prob, probs

    def _extract_activity_sequence(
        self, cat_b: List[torch.Tensor]
    ) -> List[int]:
        acts = cat_b[0].squeeze(0).cpu().tolist()
        return [a for a in acts if a != 0]

    # =========================================================================
    # Latent Space Search
    # =========================================================================

    def _search_latent_space(
        self,
        z_orig: torch.Tensor,
        original_cat: List[torch.Tensor],
        original_num_stacked: torch.Tensor,
        original_num_list: List[torch.Tensor],
        original_pred: int,
        target_class: Optional[int],
        fixed_prefix_len: int,
        actual_prefix_len: int,
    ) -> List[RevisedPlusCounterfactual]:
        """
        Iterative elite-sampling search in VAE latent space.

        The VAE was trained on random-length prefixes, so decoded outputs
        are naturally prefix-shaped (events followed by padding). No hard
        truncation needed — length changes are valid counterfactuals.

        Args:
            fixed_prefix_len: Leading positions to keep from original (immutable).
            actual_prefix_len: Real prefix length (non-padding events in input).
                              Logged for diagnostics but not used for truncation.
        """
        all_valid = []
        z_center = z_orig.clone()
        noise_scale = self.config.initial_noise_scale
        n_candidates = self.config.n_candidates_per_round
        elite_k = max(1, int(n_candidates * self.config.elite_fraction))

        for round_idx in range(self.config.n_search_rounds):
            # Sample z candidates
            z_noise = torch.randn(
                n_candidates, z_center.shape[1], device=self.device
            )
            z_candidates = z_center.expand(n_candidates, -1) + noise_scale * z_noise

            # Decode all candidates through VAE
            # The VAE was trained on prefixes, so outputs are naturally
            # prefix-shaped: real events followed by padding zeros.
            self.vae.eval()
            with torch.no_grad():
                cat_logits, num_out = self.vae.decode(z_candidates)

            # Discrete decode: argmax on categorical logits
            cat_decoded = [logits.argmax(dim=-1) for logits in cat_logits]
            # num_out: [N, seq_len, n_num]

            # Splice fixed prefix back in (positions 0..fixed_prefix_len-1)
            if fixed_prefix_len > 0:
                for i, orig_c in enumerate(original_cat):
                    orig_prefix = orig_c[:, :fixed_prefix_len].expand(n_candidates, -1)
                    cat_decoded[i] = torch.cat(
                        [orig_prefix, cat_decoded[i][:, fixed_prefix_len:]], dim=1
                    )
                if num_out is not None and original_num_stacked.shape[2] > 0:
                    orig_num_prefix = original_num_stacked[:, :fixed_prefix_len, :].expand(
                        n_candidates, -1, -1
                    )
                    num_out = torch.cat(
                        [orig_num_prefix.to(self.device), num_out[:, fixed_prefix_len:, :]],
                        dim=1,
                    )

            # Length constraint: filter candidates whose prefix length
            # deviates too much from the original
            if self.config.max_length_delta is not None:
                cf_lengths = (cat_decoded[0] != 0).sum(dim=1).cpu().numpy()
                length_ok = np.abs(cf_lengths - actual_prefix_len) <= self.config.max_length_delta
            else:
                length_ok = np.ones(n_candidates, dtype=bool)

            # Only predict candidates that pass the length filter
            keep_indices = np.where(length_ok)[0]
            if len(keep_indices) == 0:
                noise_scale = max(
                    noise_scale * self.config.noise_decay, self.config.min_noise_scale
                )
                continue

            cat_filtered = [c[keep_indices] for c in cat_decoded]
            num_filtered = num_out[keep_indices] if num_out is not None else None

            # Batch predict through U-ED-LSTM
            preds, probs = self.predictor.predict_batch(cat_filtered, num_filtered)

            # Filter valid counterfactuals
            valid_mask = preds != original_pred
            if target_class is not None:
                valid_mask &= preds == target_class

            valid_indices = np.where(valid_mask)[0]
            # Map back to z_candidates indices
            z_indices = keep_indices[valid_indices]

            # Score valid candidates
            # valid_indices indexes into cat_filtered/num_filtered/preds/probs
            # z_indices maps those back to z_candidates
            for i, filt_idx in enumerate(valid_indices):
                z_idx = z_indices[i]
                cf = self._score_candidate(
                    candidate_cat=[c[filt_idx : filt_idx + 1] for c in cat_filtered],
                    candidate_num=num_filtered[filt_idx : filt_idx + 1] if num_filtered is not None else None,
                    candidate_z=z_candidates[z_idx],
                    original_cat=original_cat,
                    original_num_stacked=original_num_stacked,
                    candidate_pred=int(preds[filt_idx]),
                    candidate_prob=float(probs[filt_idx, int(preds[filt_idx])]),
                    target_class=target_class,
                )
                if cf is not None:
                    all_valid.append(cf)

            # Refine: shift z_center toward elite candidates
            if all_valid:
                sorted_cfs = sorted(all_valid, key=lambda c: c.combined_score)
                elite_z = torch.stack(
                    [cf.z for cf in sorted_cfs[:elite_k] if cf.z is not None]
                )
                if elite_z.shape[0] > 0:
                    z_center = elite_z.mean(dim=0, keepdim=True).to(self.device)

            # Decay noise
            noise_scale = max(
                noise_scale * self.config.noise_decay, self.config.min_noise_scale
            )

        return all_valid

    # =========================================================================
    # Scoring
    # =========================================================================

    def _score_candidate(
        self,
        candidate_cat: List[torch.Tensor],
        candidate_num: Optional[torch.Tensor],
        candidate_z: torch.Tensor,
        original_cat: List[torch.Tensor],
        original_num_stacked: torch.Tensor,
        candidate_pred: int,
        candidate_prob: float,
        target_class: Optional[int],
    ) -> Optional[RevisedPlusCounterfactual]:
        """Score a single valid counterfactual candidate.

        Returns None if the candidate fails validity gates:
        - Must have at least 1 non-padding event (reject empty sequences)
        - Must pass plausibility_optimistic threshold
        """
        act_seq = self._extract_activity_sequence(candidate_cat)

        # Reject degenerate empty sequences
        if len(act_seq) == 0:
            return None

        proximity = self._compute_proximity(
            original_cat, original_num_stacked, candidate_cat, candidate_num
        )
        sparsity = self._compute_sparsity(original_cat, candidate_cat)
        feasibility = self._compute_feasibility(candidate_cat, candidate_num)
        plaus_definite, plaus_optimistic = self._compute_plausibility(act_seq)

        # Validity gate: reject if plausibility_optimistic is too low
        if plaus_optimistic < self.config.min_plausibility:
            return None

        # Effective prefix length: positions where either sequence has events
        o_act = original_cat[0].squeeze().cpu()
        c_act = candidate_cat[0].squeeze().cpu()
        effective_len = int(((o_act != 0) | (c_act != 0)).sum())

        # Combined score uses plausibility_definite (prefix-safe only)
        combined = self._compute_combined_score(
            proximity, sparsity, feasibility, plaus_definite, effective_len
        )

        pred_name = (
            self.activity_names[candidate_pred]
            if candidate_pred < len(self.activity_names)
            else str(candidate_pred)
        )

        return RevisedPlusCounterfactual(
            cat_sequence=[c.squeeze(0).cpu() for c in candidate_cat],
            num_sequence=candidate_num.squeeze(0).cpu()
            if candidate_num is not None
            else None,
            activity_sequence=act_seq,
            counterfactual_prediction=candidate_pred,
            counterfactual_prediction_name=pred_name,
            counterfactual_probability=candidate_prob,
            proximity=proximity,
            sparsity=sparsity,
            feasibility=feasibility,
            plausibility_definite=plaus_definite,
            plausibility_optimistic=plaus_optimistic,
            combined_score=combined,
            z=candidate_z.detach().cpu(),
        )

    def _compute_proximity(
        self,
        original_cat: List[torch.Tensor],
        original_num: torch.Tensor,
        candidate_cat: List[torch.Tensor],
        candidate_num: Optional[torch.Tensor],
    ) -> float:
        """Multi-feature distance: categorical L1 + numerical L2.

        Uses union mask: positions where either original or counterfactual
        has events. This correctly captures added/removed events in
        variable-length counterfactuals (left-padded data).
        """
        distance = 0.0

        # Union mask: positions where either sequence has events
        o_act = original_cat[0].squeeze().cpu()
        c_act = candidate_cat[0].squeeze().cpu()
        mask = (o_act != 0) | (c_act != 0)

        for orig, cand in zip(original_cat, candidate_cat):
            o = orig.squeeze().cpu()
            c = cand.squeeze().cpu()
            distance += float((o[mask] != c[mask]).sum())

        if (
            original_num is not None
            and candidate_num is not None
            and original_num.shape[-1] > 0
        ):
            o_num = original_num.squeeze().cpu().float()
            c_num = candidate_num.squeeze().cpu().float()
            if mask.any() and o_num.dim() > 0 and o_num.shape[0] > 0:
                diff = (o_num[mask] - c_num[mask]) ** 2
                distance += float(diff.sum().sqrt())

        return distance

    def _compute_sparsity(
        self,
        original_cat: List[torch.Tensor],
        candidate_cat: List[torch.Tensor],
    ) -> int:
        """L0: count of event positions where any categorical feature changed.

        Uses union mask to capture both original and counterfactual event
        positions (handles variable-length counterfactuals).
        """
        o_act = original_cat[0].squeeze().cpu()
        c_act = candidate_cat[0].squeeze().cpu()
        mask = (o_act != 0) | (c_act != 0)

        any_changed = torch.zeros(mask.shape, dtype=torch.bool)
        for orig, cand in zip(original_cat, candidate_cat):
            o = orig.squeeze().cpu()
            c = cand.squeeze().cpu()
            any_changed |= o != c

        return int((any_changed & mask).sum())

    def _compute_feasibility(
        self,
        candidate_cat: List[torch.Tensor],
        candidate_num: Optional[torch.Tensor],
    ) -> float:
        """VAE reconstruction loss (lower = more data-like)."""
        self.vae.eval()
        with torch.no_grad():
            cat_dev = [c.to(self.device) for c in candidate_cat]
            if candidate_num is not None and candidate_num.shape[-1] > 0:
                num_dev = candidate_num.to(self.device)
            else:
                num_dev = torch.zeros(
                    1, self.seq_len, 0, device=self.device
                )

            outputs = self.vae(cat_dev, num_dev)
            loss_dict = self.vae.loss(cat_dev, num_dev, outputs, kl_weight=0.0)
            return float(loss_dict["recon_cat"] + loss_dict["recon_num"])

    def _compute_plausibility(
        self,
        activity_seq: List[int],
    ) -> Tuple[float, float]:
        """
        Two-tier plausibility scoring.

        Returns:
            (plausibility_definite, plausibility_optimistic)
            - definite: prefix-safe constraints only (monotonic violations).
              Used as penalty in the combined score during search.
            - optimistic: all constraints, treating prefix as complete trace.
              Used as validity gate — if even this lenient check fails,
              the counterfactual is definitely implausible.
        """
        plaus_definite = DeclareConstraintChecker.satisfaction_rate(
            activity_seq, self.prefix_safe_constraints
        )
        plaus_optimistic = DeclareConstraintChecker.satisfaction_rate(
            activity_seq, self.all_constraints
        )
        return plaus_definite, plaus_optimistic

    def _compute_combined_score(
        self,
        proximity: float,
        sparsity: int,
        feasibility: float,
        plausibility_definite: float,
        prefix_len: int = 0,
    ) -> float:
        """
        Weighted combination of normalized metrics. Lower = better.

        Normalizes proximity and sparsity by the actual prefix length
        (number of event positions), not the full tensor length.
        Uses plausibility_definite (prefix-safe only) for the penalty,
        since only monotonic violations are reliable on prefixes.
        """
        effective_len = prefix_len if prefix_len > 0 else self.seq_len
        n_features = len(self.cat_vocab_sizes) + self.num_features
        norm_prox = proximity / (effective_len * n_features) if effective_len > 0 else 0.0
        norm_sparse = sparsity / effective_len if effective_len > 0 else 0.0
        norm_feas = min(feasibility, 10.0) / 10.0
        plaus_penalty = 1.0 - plausibility_definite

        return (
            self.config.w_proximity * norm_prox
            + self.config.w_sparsity * norm_sparse
            + self.config.w_feasibility * norm_feas
            + self.config.w_plausibility * plaus_penalty
        )

    # =========================================================================
    # Diversity Selection
    # =========================================================================

    def _select_diverse(
        self,
        counterfactuals: List[RevisedPlusCounterfactual],
        top_k: int,
    ) -> List[RevisedPlusCounterfactual]:
        """Greedy diversity selection: pick best, then next-best sufficiently different."""
        if not counterfactuals:
            return []

        sorted_cfs = sorted(counterfactuals, key=lambda c: c.combined_score)

        if len(sorted_cfs) <= top_k:
            return sorted_cfs

        selected = [sorted_cfs[0]]
        for cf in sorted_cfs[1:]:
            if len(selected) >= top_k:
                break
            # Check diversity against all selected
            is_diverse = True
            for sel in selected:
                # L0 ratio between the two counterfactuals' activity sequences
                max_len = max(len(cf.activity_sequence), len(sel.activity_sequence))
                if max_len == 0:
                    continue
                cf_padded = cf.activity_sequence + [0] * (
                    max_len - len(cf.activity_sequence)
                )
                sel_padded = sel.activity_sequence + [0] * (
                    max_len - len(sel.activity_sequence)
                )
                diff_ratio = sum(
                    1 for a, b in zip(cf_padded, sel_padded) if a != b
                ) / max_len
                if diff_ratio < self.config.diversity_threshold:
                    is_diverse = False
                    break
            if is_diverse:
                selected.append(cf)

        # If we couldn't fill top_k with diverse picks, add remaining best
        if len(selected) < top_k:
            for cf in sorted_cfs:
                if cf not in selected:
                    selected.append(cf)
                if len(selected) >= top_k:
                    break

        return selected


# =============================================================================
# Factory Function
# =============================================================================

def create_revised_plus_for_model(
    model: torch.nn.Module,
    dataset,
    activity_names: List[str],
    config: Optional[RevisedPlusConfig] = None,
    pretrained_vae: Optional[SimpleSequenceVAE] = None,
    vae_path: Optional[str] = None,
) -> RevisedPlus:
    """
    Convenience factory: create and fit REVISED+ from a model and dataset.

    Args:
        model: Trained U-ED-LSTM model.
        dataset: EventLogDataset with items (cat_tuple, num_tuple, case_id).
        activity_names: Activity vocabulary ordered by index.
        config: Configuration. Uses defaults if None.
        pretrained_vae: Skip VAE training if provided.
        vae_path: Path to save/load the VAE model. If the file exists,
                 the VAE is loaded instead of trained. After training,
                 the VAE is saved to this path.

    Returns:
        Fitted RevisedPlus instance ready for explain().
    """
    return RevisedPlus.from_dataset(
        model=model,
        dataset=dataset,
        activity_names=activity_names,
        config=config,
        pretrained_vae=pretrained_vae,
        vae_path=vae_path,
    )
