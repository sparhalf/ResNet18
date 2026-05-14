from __future__ import annotations

"""训练曲线、混淆矩阵、条形图等 matplotlib 导出（dpi 固定约 160 便于报告插入）。"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_curves(history: list[dict[str, Any]], out_path: Path) -> None:
    """双 subplot：train/val 的 loss 与 Top-1 accuracy 随 epoch 变化。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val")
    axes[1].set_title("Top-1 Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
    out_path: Path,
) -> None:
    """绘制混淆矩阵热力图并在格内标注样本计数。"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion matrix")

    # 每个格子标注计数；颜色与背景对比便于阅读
    vmax = float(cm.max()) if cm.size else 1.0
    thresh = 0.5 * vmax
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=10,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_per_class_bars(
    class_names: list[str],
    recalls: list[float],
    f1s: list[float],
    out_path: Path,
) -> None:
    """各类 recall 与 F1 的分组柱状图。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(class_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, recalls, width=w, label="recall")
    ax.bar(x + w / 2, f1s, width=w, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class recall & F1")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_topk_bar(topk_acc: dict[int, float], out_path: Path) -> None:
    """测试集上各 Top-k 准确率柱状图（k 为字典键）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ks = sorted(topk_acc.keys())
    vals = [topk_acc[k] for k in ks]
    labels = [f"top-{k}" for k in ks]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, vals, color="#4472c4")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Top-k accuracy (test)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
