"""TRM-faithful sparse puzzle embedding + distributed SignSGD (ARC per-(task,aug) prefix).

Ported from TinyRecursiveModels/models/sparse_embedding.py. The table lives in
nn.Buffers (not Parameters) so AdamW never sees it; gradients flow through a
per-batch `local_weights` buffer and the optimizer scatters sign-SGD updates
back into the full table (all-gather across ranks under DDP).
"""
import math
from typing import Union

import torch
from torch import nn
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, ParamsT


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # jax-style truncated normal (std is the actual std of the result)
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor


class CastedSparseEmbedding(nn.Module):
    """Sparse puzzle-embedding table with a per-step scratch buffer for the rows the
    current optimizer step touches.

    GRADIENT ACCUMULATION: local_weights.grad accumulates across all micro-batches of a
    step (zero_grad only runs after optim.step()), so the scratch buffer holds one slot
    per micro-batch, and each slot keeps the ids its gradient belongs to.

    Slots are written round-robin; the optimizer consumes the whole buffer, and duplicate
    ids across slots are folded by the unique()+scatter_add_ in _sparse_emb_signsgd_dist.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float,
                 cast_to: torch.dtype, grad_accum: int = 1):
        super().__init__()
        self.cast_to = cast_to
        self.accum = max(1, int(grad_accum))
        self.batch_size = batch_size
        self._slot = 0
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True
        )
        self.local_weights = nn.Buffer(
            torch.zeros(self.accum * batch_size, embedding_dim, requires_grad=True), persistent=False)
        # id 0 is <blank>; unused slots stay 0 and their grad stays 0, so folding them in
        # adds sign(0) = 0 to the blank row and changes nothing.
        self.local_ids = nn.Buffer(torch.zeros(self.accum * batch_size, dtype=torch.int32), persistent=False)

    def set_slot(self, i: int) -> None:
        """Choose the accumulation slot for the NEXT forward. Driven from the train loop,
        never mutated inside forward(): under torch.compile a Python-attribute mutation in
        the traced region can be dropped. Read-only here, so dynamo just guards on the
        value and keeps one compiled variant per slot."""
        self._slot = int(i) % self.accum

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.weights[inputs].to(self.cast_to)
        n = inputs.shape[0]
        lo = self._slot * self.batch_size
        with torch.no_grad():
            self.local_weights[lo:lo + n].copy_(self.weights[inputs])
            self.local_ids[lo:lo + n].copy_(inputs)
        return self.local_weights[lo:lo + n].to(self.cast_to)


class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(self, params: ParamsT, world_size: int,
                 lr: Union[float, torch.Tensor] = 1e-3, weight_decay: float = 1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, world_size=world_size))

    @torch.no_grad
    def step(self, closure=None):  # type: ignore
        for group in self.param_groups:
            local_weights_grad = local_ids = weights = None
            assert len(group["params"]) == 3
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
                else:
                    assert False
            assert local_ids is not None and weights is not None
            if local_weights_grad is not None:
                _sparse_emb_signsgd_dist(local_weights_grad, local_ids, weights,
                                         lr=group["lr"], weight_decay=group["weight_decay"],
                                         world_size=group["world_size"])


def _sparse_emb_signsgd_dist(local_weights_grad, local_ids, weights, lr, weight_decay, world_size):
    N, D = local_weights_grad.shape
    all_weights_grad, all_ids = local_weights_grad, local_ids
    if world_size > 1:
        all_weights_grad = torch.empty((world_size * N, D), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
        all_ids = torch.empty(world_size * N, dtype=local_ids.dtype, device=local_ids.device)
        dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        dist.all_gather_into_tensor(all_ids, local_ids)
    grad_ids, inv = all_ids.unique(return_inverse=True)
    grad = torch.zeros((grad_ids.shape[0], D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)
    p = weights[grad_ids]
    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)
    weights[grad_ids] = p