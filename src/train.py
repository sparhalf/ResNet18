from __future__ import annotations

"""训练与评估核心逻辑。

负责：YAML 与默认配置合并、工程路径解析、随机种子、优化器/调度器/损失函数、
单 epoch 训练循环及验证/测试用的 evaluate。日志在 TTY 上用 tqdm，否则按间隔打印行日志。
"""

import os
import random
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics.classification import accuracy_topk


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个字典；override 中的叶子覆盖 base。"""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _dict_to_namespace(d: dict[str, Any]) -> SimpleNamespace:
    """嵌套 dict → 可点号访问；list 保持原样。"""
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def _find_project_root(config_path: Path) -> Path:
    cur = config_path.resolve().parent
    for _ in range(10):
        default_yaml = cur / "configs" / "default.yaml"
        if default_yaml.is_file():
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError(
        f"找不到 configs/default.yaml（请从项目根目录附近选择配置文件）: {config_path}",
    )


def yaml_load(path: Path) -> Any:
    import yaml

    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是 mapping")
    return data


def load_config(path: str | Path) -> SimpleNamespace:
    """读取用户 YAML，并与项目内 `configs/default.yaml` 深度合并后转为可点号访问的对象。"""
    path = Path(path)
    root = _find_project_root(path)
    base_raw = yaml_load(root / "configs" / "default.yaml")
    user_raw = yaml_load(path)
    if not isinstance(user_raw, dict):
        raise ValueError("配置文件顶层必须是 YAML mapping（字典）")
    merged = _deep_merge(base_raw, user_raw)
    return _dict_to_namespace(merged)


def resolve_paths(cfg: SimpleNamespace, project_root: Path) -> tuple[Path, Path, Path]:
    """由配置解析 train 目录、test 目录与 artifacts 根目录（均为绝对路径）。"""
    data_root = (project_root / cfg.paths.data_root).resolve()
    artifacts = (project_root / cfg.paths.artifacts_dir).resolve()
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    return train_dir, test_dir, artifacts


Config = SimpleNamespace


def set_seed(seed: int) -> None:
    """固定 Python/NumPy/PyTorch 种子，并打开 cudnn 确定性（略损速度换可复现）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def build_optimizer(model: nn.Module, cfg: Config):
    """按 cfg.optimizer.name 构建 SGD / Adam / AdamW（仅 SGD 使用 momentum、nesterov）。"""
    o = cfg.optimizer
    params = model.parameters()
    name = o.name.lower()
    if name == "sgd":
        return SGD(
            params,
            lr=o.lr,
            momentum=o.momentum,
            weight_decay=o.weight_decay,
            nesterov=o.nesterov,
        )
    if name == "adam":
        return Adam(params, lr=o.lr, weight_decay=o.weight_decay)
    if name == "adamw":
        return AdamW(params, lr=o.lr, weight_decay=o.weight_decay)
    raise ValueError(f"unsupported optimizer: {o.name}")


def build_scheduler(optimizer, cfg: Config):
    """学习率调度器工厂：cosine / step / multistep / none。"""
    s = cfg.scheduler
    name = s.name.lower()
    if name == "none" or name == "null":
        return None
    if name == "cosine":
        T_max = s.T_max if s.T_max is not None else cfg.train.epochs
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=s.eta_min)
    if name == "step":
        return StepLR(optimizer, step_size=s.step_size, gamma=s.gamma)
    if name == "multistep":
        return MultiStepLR(optimizer, milestones=list(s.milestones), gamma=s.gamma)
    raise ValueError(f"unsupported scheduler: {s.name}")


def build_criterion(cfg: Config) -> nn.Module:
    """多分类交叉熵，支持标签平滑（见 cfg.train.label_smoothing）。"""
    return nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)


def _stderr_tty() -> bool:
    return sys.stderr.isatty()


def _train_log_every_n(cfg_logging) -> int:
    n = getattr(cfg_logging, "train_log_every_n_batches", None)
    if n is not None:
        return max(1, int(n))
    return max(1, int(getattr(cfg_logging, "log_interval", 1)))


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    topk: tuple[int, ...],
    epoch: int,
    num_epochs: int,
    cfg_logging,
    grad_clip_norm: float,
) -> tuple[float, float, dict[int, float]]:
    """单轮训练：前向、反向、可选梯度裁剪；返回本 epoch 平均 loss、Top-1 acc、各 Top-k acc。"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    topk_correct = {k: 0 for k in topk}
    every_n = _train_log_every_n(cfg_logging)
    use_tqdm = _stderr_tty()

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=f"Ep {epoch}/{num_epochs} train",
        leave=False,
        disable=not use_tqdm,
        dynamic_ncols=True,
        mininterval=0.5 if use_tqdm else 10**9,
    )

    for batch_idx, (xb, yb) in enumerate(pbar, start=1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        bs = xb.size(0)
        loss_val = loss.item()
        total_loss += loss_val * bs
        pred = logits.argmax(dim=1)
        batch_correct = (pred == yb).sum().item()
        total_correct += batch_correct
        total += bs
        for k in topk:
            topk_correct[k] += accuracy_topk(logits, yb, k) * bs

        avg_loss = total_loss / max(1, total)
        batch_acc = batch_correct / max(1, bs)
        running_acc = total_correct / max(1, total)
        lr = optimizer.param_groups[0]["lr"]

        topk_parts = {k: accuracy_topk(logits, yb, k) for k in topk}

        if use_tqdm:
            post = {
                "loss": f"{loss_val:.4f}",
                "avgL": f"{avg_loss:.4f}",
                "bAcc": f"{batch_acc:.3f}",
                "rAcc": f"{running_acc:.3f}",
                "lr": f"{lr:.2e}",
            }
            for k in topk:
                post[f"t{k}"] = f"{topk_parts[k]:.3f}"
            pbar.set_postfix(post, refresh=True)
        elif batch_idx % every_n == 0 or batch_idx == len(loader):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tk = " ".join(f"top{k}={topk_parts[k]:.4f}" for k in topk)
            print(
                f"[{ts}] "
                f"epoch {epoch:03d}/{num_epochs} "
                f"step {batch_idx:04d}/{len(loader)} "
                f"loss={loss_val:.6f} "
                f"avg_loss={avg_loss:.6f} "
                f"batch_acc={batch_acc:.4f} "
                f"run_acc={running_acc:.4f} "
                f"lr={lr:.6e} "
                f"{tk}",
                flush=True,
            )

    if use_tqdm:
        pbar.close()

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    topk_acc = {k: topk_correct[k] / max(1, total) for k in topk}
    return avg_loss, acc, topk_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    topk: tuple[int, ...],
    desc: str = "eval",
) -> tuple[float, float, dict[int, float]]:
    """在 loader 上评估（无梯度）：平均 loss、Top-1、各 Top-k。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    topk_correct = {k: 0 for k in topk}
    use_tqdm = _stderr_tty()

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        leave=False,
        disable=not use_tqdm,
        dynamic_ncols=True,
        mininterval=0.3 if use_tqdm else 10**9,
    )

    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        pred = logits.argmax(dim=1)
        total_correct += (pred == yb).sum().item()
        total += bs
        for k in topk:
            topk_correct[k] += accuracy_topk(logits, yb, k) * bs

        if use_tqdm:
            avg_loss = total_loss / max(1, total)
            acc = total_correct / max(1, total)
            post = {"loss": f"{loss.item():.4f}", "avgL": f"{avg_loss:.4f}", "acc": f"{acc:.3f}"}
            for k in topk:
                post[f"t{k}"] = f"{topk_correct[k] / max(1, total):.3f}"
            pbar.set_postfix(post, refresh=False)

    if use_tqdm:
        pbar.close()

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    topk_acc = {k: topk_correct[k] / max(1, total) for k in topk}
    return avg_loss, acc, topk_acc
