from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class FlowLossHead(nn.Module):
    """CE on the decoded x1_pred of the Euler rollout.

    Train: one rollout at train_N (stochastic time grid), CE against labels.
    Eval: Euler sweep over eval_Ns (uniform grids); metrics get an _N{N} suffix,
    preds/loss come from the largest N.
    """

    def __init__(self, model: nn.Module, loss_type: str, label_smoothing: float = 0.0):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.loss_fn = globals()[loss_type]
        self.label_smoothing = float(label_smoothing)

    def _smoothed_ce(self, logits: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # CE of the smoothed one-hot target against the log-probs
        # (stablemax in f64 or softmax in f32, matching loss_type)
        if self.loss_type == "stablemax_cross_entropy":
            logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
        else:
            logprobs = F.log_softmax(logits.to(torch.float32), dim=-1)
        vocab = logits.shape[-1]
        eps = self.label_smoothing
        target = F.one_hot(torch.where(valid_mask, labels, 0).to(torch.long), vocab).to(logprobs.dtype)
        target = (1.0 - eps) * target + eps / vocab
        ce = -(target * logprobs).sum(-1)
        return torch.where(valid_mask, ce, 0)

    def _curriculum_mask(self, inputs: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Stage k of curr_stages: reveal a row-major-ordered k/curr_stages fraction of the
        "reasoning" cells -- defined as inputs != labels (blank Sudoku cells; maze path
        cells; dataset-agnostic, no vocab hardcoding) -- cumulatively, PLUS all "given"
        cells (inputs == labels), which are free and never need staging. Same target
        semantics as the ProsQA staged curriculum (prosqa/flow.py), ported to a grid instead
        of a graph path. Not used by the paper configs (curriculum="none"). See the docstring
        on DiTRefConfig.curriculum for why this is NOT deep
        supervision (one loss per rollout throughout; only which positions count toward
        it changes, and only across epochs)."""
        cfg = self.model.config
        diff = (inputs != labels) & valid_mask
        given = valid_mask & ~diff
        stage = float(getattr(self.model, "curr_stage", cfg.curr_stages))
        rank = torch.cumsum(diff.long(), dim=-1)                      # 1-indexed within diff cells
        n_diff = diff.sum(-1, keepdim=True).float()
        threshold = torch.ceil(stage / cfg.curr_stages * n_diff).long()
        revealed = diff & (rank <= threshold)
        return given | revealed

    def _loss_and_metrics(self, logits: torch.Tensor, labels: torch.Tensor, suffix: str = "",
                           inputs: torch.Tensor = None):
        mask = (labels != IGNORE_LABEL_ID)
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            # accuracy/exact_accuracy always use the FULL mask -- curriculum restricts the
            # LOSS only, never the reported metrics, so eval and checkpoint-selection
            # accuracy are never curriculum-inflated (forward_eval doesn't reach this
            # method's curriculum branch at all: self.training is False there).
            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            valid_metrics = loss_counts > 0

            metrics = {
                f"accuracy{suffix}":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                f"exact_accuracy{suffix}": (valid_metrics & seq_is_correct).sum(),
            }

        loss_mask, loss_divisor_l = mask, loss_divisor
        cfg = getattr(self.model, "config", None)
        if self.training and inputs is not None and cfg is not None and getattr(cfg, "curriculum", "none") == "staged":
            loss_mask = self._curriculum_mask(inputs, labels, mask)
            loss_divisor_l = loss_mask.sum(-1).clamp_min(1).unsqueeze(-1)

        if self.label_smoothing > 0.0:
            lm_loss = (self._smoothed_ce(logits, labels, valid_mask=loss_mask) / loss_divisor_l).sum()
        else:
            lm_loss = (self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=loss_mask) / loss_divisor_l).sum()
        metrics[f"lm_loss{suffix}"] = lm_loss.detach()
        return lm_loss, metrics, preds, valid_metrics

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_keys: Sequence[str],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        labels = batch["labels"]

        if self.training:
            logits = self.model(batch)
            idx = getattr(self.model, "_crop_idx", None)
            inputs = batch["inputs"]
            if idx is not None:
                labels = labels[:, idx]
                inputs = inputs[:, idx]
            loss, metrics, preds, valid_metrics = self._loss_and_metrics(logits, labels, inputs=inputs)
            aux = getattr(self.model, "_aux_loss", None)
            if aux is not None:
                loss = loss + aux * valid_metrics.sum()  # scale like the summed CE
                self.model._aux_loss = None
            outputs = {"logits": logits, "preds": preds}
        else:
            logits_per_N = self.model.forward_eval(batch)
            idx = getattr(self.model, "_crop_idx", None)
            if idx is not None:
                labels = labels[:, idx]
            loss = None
            metrics = {}
            outputs = {}
            for N, logits in logits_per_N.items():
                loss, m, preds, valid_metrics = self._loss_and_metrics(logits, labels, suffix=f"_N{N}")
                metrics.update(m)
                # largest N (last key) wins for preds/logits/loss
                outputs = {"logits": logits, "preds": preds}

        metrics["count"] = valid_metrics.sum()

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return loss, metrics, detached_outputs
