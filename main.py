from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset
from src.train import (
    Config,
    build_criterion,
    build_optimizer,
    build_scheduler,
    evaluate,
    load_config,
    resolve_paths,
    set_seed,
    train_epoch,
)
from src.data.dataset import FilteredImageFolder, load_or_create_split
from src.data.transforms import build_transforms
from src.metrics.classification import gather_predictions
from src.metrics.gradcam import save_gradcam_grid
from src.metrics.plots import (
    plot_confusion,
    plot_curves,
    plot_per_class_bars,
    plot_topk_bar,
)
from src.models import build_resnet


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _device(cfg_pin_memory: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_loaders(cfg: Config, train_dir: Path, artifacts: Path):
    split_dir = artifacts / "splits"
    train_idx, val_idx = load_or_create_split(
        train_root=train_dir,
        val_ratio=cfg.repro.val_ratio,
        seed=cfg.repro.seed,
        cache_dir=split_dir,
        use_cache=cfg.repro.split_cache,
    )

    img_sz = int(getattr(cfg.data, "image_size", 96))
    aug_on = bool(getattr(cfg.data, "augmentation_enabled", True))
    if aug_on:
        train_tf = build_transforms(
            cfg.data.augmentation_train,
            image_size=img_sz,
            ensure_min_train_aug=True,
        )
    else:
        train_tf = build_transforms(
            [],
            image_size=img_sz,
            ensure_min_train_aug=False,
        )
    val_tf = build_transforms(cfg.data.augmentation_val_test)

    train_base = FilteredImageFolder(str(train_dir), transform=train_tf)
    val_base = FilteredImageFolder(str(train_dir), transform=val_tf)

    train_loader = DataLoader(
        Subset(train_base, train_idx),
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    val_loader = DataLoader(
        Subset(val_base, val_idx),
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    return train_loader, val_loader, train_base.classes


def cmd_train(config_path: Path) -> None:
    cfg = load_config(config_path)
    root = _project_root()
    set_seed(cfg.repro.seed)
    train_dir, _test_dir, artifacts = resolve_paths(cfg, root)
    artifacts.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, artifacts / "config_used.yaml")

    device = _device(cfg.data.pin_memory)
    train_loader, val_loader, classes = _make_loaders(cfg, train_dir, artifacts)

    model = build_resnet(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = build_criterion(cfg)

    topk = tuple(sorted(set(cfg.metrics.topk)))
    history: list[dict] = []
    best_metric = float("-inf") if cfg.logging.save_best_by == "val_acc" else float("inf")
    best_path = artifacts / "best.pt"

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    banner = (
        f"\n{'=' * 78}\n"
        f"  STL-10 training started  {ts}\n"
        f"  config     : {config_path}\n"
        f"  device     : {device}\n"
        f"  artifacts  : {artifacts}\n"
        f"  train/val  : {n_train} / {n_val} samples  |  batch_size={cfg.data.batch_size}\n"
        f"  epochs     : {cfg.train.epochs}  |  topk={list(topk)}\n"
        f"{'=' * 78}\n"
    )
    print(banner, flush=True)

    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss, tr_acc, tr_topk = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            topk,
            epoch,
            cfg.train.epochs,
            cfg.logging,
            cfg.train.grad_clip_norm,
        )
        va_loss, va_acc, va_topk = evaluate(
            model,
            val_loader,
            criterion,
            device,
            topk,
            desc="val",
        )
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_acc": tr_acc,
            "val_acc": va_acc,
        }
        for k in topk:
            row[f"train_top{k}"] = tr_topk[k]
            row[f"val_top{k}"] = va_topk[k]
        history.append(row)

        if cfg.logging.save_best_by == "val_acc":
            improved = va_acc > best_metric
            if improved:
                best_metric = va_acc
        else:
            improved = va_loss < best_metric
            if improved:
                best_metric = va_loss

        if improved:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": va_acc,
                    "val_loss": va_loss,
                    "classes": classes,
                },
                best_path,
            )

        lr_now = optimizer.param_groups[0]["lr"]
        tk_tr = " ".join(f"train_top{k}={tr_topk[k]:.4f}" for k in topk)
        tk_va = " ".join(f" val_top{k}={va_topk[k]:.4f}" for k in topk)
        star = "*" * 78
        summary = (
            f"\n{star}\n"
            f"  Epoch {epoch:03d}/{cfg.train.epochs}  summary  |  lr={lr_now:.6e}\n"
            f"  {'-' * 74}\n"
            f"  Train  loss={tr_loss:.6f}  acc={tr_acc:.6f}  |  {tk_tr}\n"
            f"  Val    loss={va_loss:.6f}  acc={va_acc:.6f}  |  {tk_va}\n"
            f"  {'-' * 74}\n"
            f"  monitor={cfg.logging.save_best_by}  best={best_metric:.6f}  "
            f"{'* checkpoint updated' if improved else '(unchanged)'}\n"
            f"{star}\n"
        )
        print(summary, flush=True)

    with open(artifacts / "metrics_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_curves(history, artifacts / "curves_loss_acc.png")
    done_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"\n{'=' * 78}\n"
        f"  Training finished  {done_ts}\n"
        f"  metrics_history.json / curves_loss_acc.png / best.pt -> {artifacts}\n"
        f"{'=' * 78}\n",
        flush=True,
    )

    if getattr(cfg.logging, "eval_on_test_after_train", True):
        print(
            "\n>>> logging.eval_on_test_after_train=true：在测试集上生成测评文件 …\n",
            flush=True,
        )
        cmd_eval_test(config_path)
    else:
        print(
            "\n>>> 未自动测评测试集。请手动执行：\n"
            f"    python main.py eval_test --config {config_path}\n",
            flush=True,
        )

    if getattr(cfg.gradcam, "run_after_train", True):
        print(
            "\n>>> gradcam.run_after_train=true：生成 Grad-CAM 可视化 …\n",
            flush=True,
        )
        cmd_gradcam(config_path)


def cmd_eval_test(config_path: Path) -> None:
    cfg = load_config(config_path)
    root = _project_root()
    set_seed(cfg.repro.seed)
    train_dir, test_dir, artifacts = resolve_paths(cfg, root)
    ckpt_path = artifacts / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    device = _device(cfg.data.pin_memory)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]

    val_tf = build_transforms(cfg.data.augmentation_val_test)
    test_ds = FilteredImageFolder(str(test_dir), transform=val_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    model = build_resnet(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state"])

    criterion = build_criterion(cfg)
    topk = tuple(sorted(set(cfg.metrics.topk)))
    loss, acc, topk_acc = evaluate(
        model, test_loader, criterion, device, topk, desc="test"
    )

    y_true, y_pred = gather_predictions(model, test_loader, device)
    label_ids = list(range(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=classes,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    artifacts.mkdir(parents=True, exist_ok=True)
    text_lines = [f"test_loss={loss:.6f}", f"test_top1_acc={acc:.6f}"]
    for k in topk:
        text_lines.append(f"test_top{k}_acc={topk_acc[k]:.6f}")
    text_lines.append("")
    text_lines.append(
        classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=classes,
            digits=4,
            zero_division=0,
        )
    )
    (artifacts / "test_report.txt").write_text("\n".join(text_lines), encoding="utf-8")

    plot_confusion(y_true, y_pred, classes, artifacts / "test_confusion_matrix.png")

    recalls = [report[c]["recall"] for c in classes]
    f1s = [report[c]["f1-score"] for c in classes]
    plot_per_class_bars(classes, recalls, f1s, artifacts / "test_per_class_recall_f1.png")

    plot_topk_bar(topk_acc, artifacts / "test_topk_bar.png")

    print("\n".join(text_lines[: min(20, len(text_lines))]))


def cmd_gradcam(config_path: Path) -> None:
    cfg = load_config(config_path)
    root = _project_root()
    set_seed(cfg.repro.seed)
    _train_dir, test_dir, artifacts = resolve_paths(cfg, root)
    ckpt_path = artifacts / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    device = _device(cfg.data.pin_memory)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]

    model = build_resnet(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state"])

    extensions = (".png", ".jpg", ".jpeg")
    candidates: list[Path] = []
    for cls_dir in sorted([p for p in test_dir.iterdir() if p.is_dir()]):
        for p in cls_dir.iterdir():
            if (
                p.is_file()
                and p.suffix.lower() in extensions
                and p.stat().st_size > 0
            ):
                candidates.append(p)

    rng = random.Random(cfg.repro.seed)
    rng.shuffle(candidates)
    paths = candidates[: cfg.gradcam.num_samples]

    preprocess = build_transforms(cfg.data.augmentation_val_test)
    out = artifacts / "gradcam_samples.png"
    save_gradcam_grid(
        model=model,
        image_paths=paths,
        class_names=classes,
        preprocess=preprocess,
        device=device,
        target_layer_name=cfg.gradcam.target_layer,
        out_path=out,
        alpha=cfg.gradcam.alpha,
    )
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="STL-10 manual ResNet trainer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train with train/val split")
    p_train.add_argument("--config", type=Path, required=True)

    p_eval = sub.add_parser("eval_test", help="final test evaluation")
    p_eval.add_argument("--config", type=Path, required=True)

    p_gc = sub.add_parser("gradcam", help="Grad-CAM on random test images")
    p_gc.add_argument("--config", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args.config)
    elif args.command == "eval_test":
        cmd_eval_test(args.config)
    elif args.command == "gradcam":
        cmd_gradcam(args.config)


if __name__ == "__main__":
    main()
