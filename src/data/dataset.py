from __future__ import annotations

"""STL-10 ImageFolder 数据集与训练/验证索引。

FilteredImageFolder：在 torchvision ImageFolder 基础上过滤无效文件。
划分函数按类别分层抽样，保证各类验证比例大致一致。
"""

import json
from pathlib import Path

import numpy as np
from torchvision.datasets import ImageFolder


class FilteredImageFolder(ImageFolder):
    """丢弃 0 字节等无效样本，避免读图报错；后续划分与 `classes` 均基于过滤后的列表。"""

    def __init__(self, root: str, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        pairs = [(p, t) for p, t in self.samples if Path(p).stat().st_size > 0]
        self.samples = pairs
        self.imgs = self.samples
        self.targets = [t for _, t in self.samples]


def class_names(train_root: Path) -> list[str]:
    """仅扫描 train 根目录下列文件夹名作为类别顺序（与 ImageFolder 一致）。"""
    ds = FilteredImageFolder(str(train_root))
    return ds.classes


def stratified_train_val_indices(
    train_root: Path,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """按类分层：每类随机打乱后前 `val_ratio` 比例进验证集，其余进训练集。"""
    ds = FilteredImageFolder(str(train_root))
    targets = np.array(ds.targets)
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        val_indices.extend(idx[:n_val].tolist())
        train_indices.extend(idx[n_val:].tolist())
    return train_indices, val_indices


def load_or_create_split(
    train_root: Path,
    val_ratio: float,
    seed: int,
    cache_dir: Path,
    use_cache: bool,
) -> tuple[list[int], list[int]]:
    """读取或生成 train/val 索引；若开启缓存且 meta 一致则直接读 JSON。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / "train_indices.json"
    val_path = cache_dir / "val_indices.json"
    meta_path = cache_dir / "split_meta.json"

    meta = {"val_ratio": val_ratio, "seed": seed}
    if (
        use_cache
        and train_path.is_file()
        and val_path.is_file()
        and meta_path.is_file()
    ):
        cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached_meta == meta:
            train_indices = json.loads(train_path.read_text(encoding="utf-8"))
            val_indices = json.loads(val_path.read_text(encoding="utf-8"))
            return train_indices, val_indices

    train_indices, val_indices = stratified_train_val_indices(
        train_root, val_ratio, seed
    )
    train_path.write_text(json.dumps(train_indices), encoding="utf-8")
    val_path.write_text(json.dumps(val_indices), encoding="utf-8")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return train_indices, val_indices
