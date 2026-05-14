from __future__ import annotations

"""Grad-CAM：对指定卷积层注册 hook，取目标 logit 对特征图的梯度做加权求和。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


def _denormalize(tensor_chw: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor_chw.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor_chw.device).view(3, 1, 1)
    img = tensor_chw * std + mean
    img = img.clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
    return img


class GradCAM:
    """Grad-CAM：在 `target_layer` 上累积前向激活与反向梯度，生成类显著性图。"""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list = []

        def fwd_hook(_module, _inp, out):
            self.activations = out.detach()

        def full_bwd_hook(_module, _grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._handles.append(target_layer.register_forward_hook(fwd_hook))
        self._handles.append(target_layer.register_full_backward_hook(full_bwd_hook))

    def close(self):
        """移除已注册的 forward/backward hook，避免内存泄漏或重复触发。"""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __call__(self, input_batch: torch.Tensor, class_idx: int | None = None):
        """对 `input_batch[0]` 计算 CAM；`class_idx` 默认取当前预测 argmax。"""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_batch)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1)[0].item())
        score = logits[0, class_idx]
        score.backward(retain_graph=False)
        assert self.activations is not None and self.gradients is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, class_idx


def overlay_cam_on_image(
    image_hwc: np.ndarray, cam_11hw: torch.Tensor, alpha: float
) -> np.ndarray:
    """将 CAM 双线性插值到图像尺寸，以 jet 伪彩与 RGB 原图按 alpha 混合。"""
    h, w = image_hwc.shape[:2]
    cam = F.interpolate(cam_11hw, size=(h, w), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    heatmap = plt.cm.jet(cam)[:, :, :3]
    out = (1 - alpha) * image_hwc + alpha * heatmap
    return np.clip(out, 0, 1)


def save_gradcam_grid(
    model: torch.nn.Module,
    image_paths: list[Path],
    class_names: list[str],
    preprocess,
    device: torch.device,
    target_layer_name: str,
    out_path: Path,
    alpha: float,
) -> None:
    """多行三列拼图：原图、热力图、叠加图；默认解释模型对 argmax 类别的归因。"""
    layer_map = {
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
    }
    if target_layer_name not in layer_map:
        raise ValueError(f"unknown layer {target_layer_name}")
    cam_engine = GradCAM(model, layer_map[target_layer_name])

    to_tensor = T.Compose([T.Resize((96, 96)), T.ToTensor()])
    fig, axes = plt.subplots(len(image_paths), 3, figsize=(9, 3 * len(image_paths)))
    if len(image_paths) == 1:
        axes = np.expand_dims(axes, axis=0)

    model.eval()
    try:
        for i, pth in enumerate(image_paths):
            pil = Image.open(pth).convert("RGB")
            raw = to_tensor(pil)
            input_tensor = preprocess(pil).unsqueeze(0).to(device)
            cam, pred_idx = cam_engine(input_tensor)
            img_np = _denormalize(input_tensor[0])
            overlay = overlay_cam_on_image(img_np, cam, alpha=alpha)

            axes[i, 0].imshow(raw.permute(1, 2, 0).numpy())
            axes[i, 0].set_title("input")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(cam[0, 0].detach().cpu(), cmap="jet")
            axes[i, 1].set_title("heatmap")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f"pred={class_names[pred_idx]}")
            axes[i, 2].axis("off")
    finally:
        cam_engine.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
