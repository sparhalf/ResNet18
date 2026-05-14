from __future__ import annotations

"""分类指标：Top-k 正确率与整集预测收集（用于 sklearn 报告与绘图）。"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, k: int) -> float:
    """batch 内 Top-k 命中率（任一 top 预测与标签相等即计为正确）；k 大于类别数时按类别数截断。"""
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
    """遍历 loader，返回 (真实标签列表, argmax 预测列表)。"""
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
