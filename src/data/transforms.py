from __future__ import annotations

"""将 YAML 中的增强描述转为 torchvision.transforms.Compose。

增强列表 + ToTensor（+ 可选 ImageNet 归一化）。训练阶段可强制补全最低限度随机增强。
"""

from typing import Any

from torchvision import transforms


def _build_one(item: dict[str, Any]):
    """将单条 `{"type": ...}` 配置映射为对应 torchvision 变换实例。"""
    t = item["type"]
    if t == "RandomHorizontalFlip":
        return transforms.RandomHorizontalFlip(p=float(item.get("p", 0.5)))
    if t == "RandomVerticalFlip":
        return transforms.RandomVerticalFlip(p=float(item.get("p", 0.5)))
    if t == "RandomCrop":
        size = item["size"]
        if isinstance(size, list):
            size = tuple(size)
        return transforms.RandomCrop(
            size,
            padding=int(item.get("padding", 0)),
            pad_if_needed=bool(item.get("pad_if_needed", False)),
        )
    if t == "RandomResizedCrop":
        size = item["size"]
        if isinstance(size, list):
            size = tuple(size)
        return transforms.RandomResizedCrop(
            size,
            scale=tuple(item.get("scale", (0.8, 1.0))),
            ratio=tuple(item.get("ratio", (0.75, 4.0 / 3.0))),
        )
    if t == "ColorJitter":
        return transforms.ColorJitter(
            brightness=item.get("brightness", 0),
            contrast=item.get("contrast", 0),
            saturation=item.get("saturation", 0),
            hue=item.get("hue", 0),
        )
    if t == "RandomRotation":
        return transforms.RandomRotation(degrees=float(item["degrees"]))
    if t == "GaussianBlur":
        return transforms.GaussianBlur(
            kernel_size=int(item.get("kernel_size", 3)),
            sigma=tuple(item.get("sigma", (0.1, 2.0))),
        )
    raise ValueError(f"unknown augmentation type: {t}")


def _ensure_min_train_augmentation(
    specs: list[dict[str, Any]],
    image_size: int,
) -> list[dict[str, Any]]:
    """至少包含 RandomHorizontalFlip 与 RandomCrop。"""
    out = list(specs)
    present = {s.get("type") for s in out}
    if "RandomHorizontalFlip" not in present:
        out.insert(0, {"type": "RandomHorizontalFlip", "p": 0.5})
        present.add("RandomHorizontalFlip")
    if "RandomCrop" not in present:
        pad = max(4, min(16, image_size // 12))
        out.append(
            {"type": "RandomCrop", "size": [image_size, image_size], "padding": pad}
        )
    return out


def build_transforms(
    specs: list[dict[str, Any]],
    *,
    normalize: bool = True,
    image_size: int = 96,
    ensure_min_train_aug: bool = False,
):
    """由 YAML 列表构建 Compose；末尾固定为 ToTensor +（可选）Normalize。"""
    s = _ensure_min_train_augmentation(specs, image_size) if ensure_min_train_aug else list(specs)
    ops = [_build_one(x) for x in s]
    base = [transforms.ToTensor()]
    if normalize:
        base.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    return transforms.Compose(ops + base)
