"""
Integrated Gradients implementation for the suffix prediction model.

Integrated Gradients (Sundararajan et al., 2017) is an attribution method that
computes the contribution of each input feature to the model's prediction.
The method satisfies important axioms like sensitivity and implementation invariance.

For sequences, attributions are computed for each input timestep and feature.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple, List, Callable, Union
import numpy as np


class IntegratedGradients:
    """
    Computes Integrated Gradients attributions for sequence-to-sequence models.

    The attribution for input feature x_i is computed as:
    IG(x_i) = (x_i - x'_i) * integral_{alpha=0}^{1} dF(x' + alpha*(x - x')) / dx_i d_alpha

    where x' is the baseline input and F is the model function.
    """

    def __init__(
        self,
        forward_fn: Callable,
        multiply_by_inputs: bool = True
    ):
        """
        Initialize the Integrated Gradients calculator.

        Args:
            forward_fn: A function that takes embedded inputs and returns the target scalar.
                       Must be differentiable with respect to its inputs.
            multiply_by_inputs: If True, multiply gradients by (input - baseline).
                               This gives the true IG attribution. Default True.
        """
        self.forward_fn = forward_fn
        self.multiply_by_inputs = multiply_by_inputs

    def compute(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Tensor, ...],
        n_steps: int = 50,
        method: str = 'riemann_trapezoid',
        return_convergence_delta: bool = False
    ) -> Union[Tuple[Tensor, ...], Tuple[Tuple[Tensor, ...], float]]:
        """
        Compute Integrated Gradients attributions.

        Args:
            inputs: Tuple of input tensors (embedded form). Each tensor requires grad.
            baselines: Tuple of baseline tensors with same shapes as inputs.
            n_steps: Number of steps for the integral approximation.
            method: Integration method - 'riemann_left', 'riemann_right',
                   'riemann_middle', or 'riemann_trapezoid'.
            return_convergence_delta: If True, return the completeness check delta.

        Returns:
            Tuple of attribution tensors with same shapes as inputs.
            If return_convergence_delta is True, also returns the convergence delta.
        """
        # Validate inputs
        assert len(inputs) == len(baselines), \
            f"Number of inputs ({len(inputs)}) must match baselines ({len(baselines)})"

        for i, (inp, base) in enumerate(zip(inputs, baselines)):
            assert inp.shape == base.shape, \
                f"Input {i} shape {inp.shape} must match baseline shape {base.shape}"

        # Get alphas for the integration path
        alphas = self._get_alphas(n_steps, method)

        # Compute gradients at each step
        all_grads = [[] for _ in inputs]

        for alpha in alphas:
            # Interpolate between baseline and input
            scaled_inputs = tuple(
                base + alpha * (inp - base)
                for inp, base in zip(inputs, baselines)
            )

            # Ensure gradients are tracked
            scaled_inputs = tuple(
                t.clone().requires_grad_(True) for t in scaled_inputs
            )

            # Forward pass and get target output
            output = self.forward_fn(*scaled_inputs)

            # Backward pass to get gradients
            grads = torch.autograd.grad(
                outputs=output,
                inputs=scaled_inputs,
                grad_outputs=torch.ones_like(output),
                retain_graph=False,
                create_graph=False
            )

            for i, grad in enumerate(grads):
                all_grads[i].append(grad.detach())

        # Integrate gradients
        integrated_grads = []
        for i, grads_list in enumerate(all_grads):
            # Stack and integrate
            stacked = torch.stack(grads_list, dim=0)  # [n_steps, ...]
            integrated = self._integrate(stacked, method, alphas)
            integrated_grads.append(integrated)

        # Multiply by (input - baseline) if requested
        if self.multiply_by_inputs:
            attributions = tuple(
                ig * (inp - base)
                for ig, inp, base in zip(integrated_grads, inputs, baselines)
            )
        else:
            attributions = tuple(integrated_grads)

        if return_convergence_delta:
            delta = self._compute_convergence_delta(
                inputs, baselines, attributions
            )
            return attributions, delta

        return attributions

    def _get_alphas(self, n_steps: int, method: str) -> List[float]:
        """Get the alpha values for integration based on the method."""
        if method == 'riemann_left':
            return [i / n_steps for i in range(n_steps)]
        elif method == 'riemann_right':
            return [(i + 1) / n_steps for i in range(n_steps)]
        elif method == 'riemann_middle':
            return [(i + 0.5) / n_steps for i in range(n_steps)]
        elif method == 'riemann_trapezoid':
            return [i / n_steps for i in range(n_steps + 1)]
        else:
            raise ValueError(f"Unknown integration method: {method}")

    def _integrate(
        self,
        grads: Tensor,
        method: str,
        alphas: List[float]
    ) -> Tensor:
        """Integrate the gradients along the path."""
        if method == 'riemann_trapezoid':
            # Trapezoidal rule: (f(0) + 2*f(1) + ... + 2*f(n-1) + f(n)) / (2*n)
            weights = torch.ones(len(alphas), device=grads.device)
            weights[0] = 0.5
            weights[-1] = 0.5
            # Reshape weights for broadcasting
            for _ in range(grads.dim() - 1):
                weights = weights.unsqueeze(-1)
            integrated = (weights * grads).sum(dim=0) / (len(alphas) - 1)
        else:
            # Riemann sum: average of all gradients
            integrated = grads.mean(dim=0)

        return integrated

    def _compute_convergence_delta(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Tensor, ...],
        attributions: Tuple[Tensor, ...]
    ) -> float:
        """
        Compute the convergence delta (completeness axiom check).

        The sum of attributions should equal F(x) - F(x').
        Delta = |F(x) - F(x') - sum(attributions)|
        """
        with torch.no_grad():
            # Get output at input
            inputs_nograd = tuple(t.detach() for t in inputs)
            output_input = self.forward_fn(*inputs_nograd)

            # Get output at baseline
            baselines_nograd = tuple(t.detach() for t in baselines)
            output_baseline = self.forward_fn(*baselines_nograd)

            # Compute sum of attributions
            total_attribution = sum(attr.sum() for attr in attributions)

            # Compute delta
            delta = abs(
                (output_input - output_baseline).sum().item() -
                total_attribution.item()
            )

        return delta
