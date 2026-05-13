from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, k: int) -> float:
    maxk = min(k, logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return correct.any(dim=1).float().mean().item()


@torch.no_grad()
def gather_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    ys: list[int] = []
    preds: list[int] = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().tolist()
        preds.extend(pred)
        ys.extend(yb.tolist())
    return ys, preds
