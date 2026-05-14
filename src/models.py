from __future__ import annotations

"""手写 ResNet-18 风格网络（BasicBlock + 四阶段堆叠）。

与 torchvision 实现细节不完全一致：stem 为 3×3 conv（适合 STL-10 等小输入），可选 stem 后 MaxPool；
分类头前为自适应全局池化（avg 或 max）+ 可选 Dropout + Linear。
"""

from types import SimpleNamespace

import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    n = name.lower()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "gelu":
        return nn.GELU()
    if n == "silu" or n == "swish":
        return nn.SiLU(inplace=True)
    if n == "tanh":
        return nn.Tanh()
    if n == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"unsupported activation: {name}")


def _get_adaptive_head_pooling(kind: str) -> nn.Module:
    k = kind.lower()
    if k == "avg":
        return nn.AdaptiveAvgPool2d((1, 1))
    if k == "max":
        return nn.AdaptiveMaxPool2d((1, 1))
    raise ValueError(f"unsupported head pooling: {kind}")


class BasicBlock(nn.Module):
    """标准两卷积残差块；stride 或通道变化时使用 1×1 shortcut 对齐尺寸。"""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        activation: nn.Module,
        norm_layer,
        use_bn: bool,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(planes) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes) if use_bn else nn.Identity()
        self.act = activation

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(planes) if use_bn else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.act(out)
        return out


class ResNetManual(nn.Module):
    """可配置激活、BN、Dropout、池化类型的 ResNet 风格分类网络。"""

    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg
        act = _get_activation(cfg.activation)

        def norm_fn(c: int):
            return nn.BatchNorm2d(c)

        self.use_bn = cfg.use_bn

        stem_layers: list[nn.Module] = [
            nn.Conv2d(
                3,
                cfg.base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            norm_fn(cfg.base_channels) if cfg.use_bn else nn.Identity(),
            act,
        ]
        # 与 torchvision ResNet 一致：stem 后可选 3×3 maxpool stride=2（小图默认关）
        if bool(getattr(cfg, "stem_maxpool", False)):
            stem_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stem = nn.Sequential(*stem_layers)
        self.in_planes = cfg.base_channels

        stage_channels = [cfg.base_channels * (2**i) for i in range(len(cfg.layers))]
        strides = [1, 2, 2, 2]
        self.layer1 = self._make_layer(
            stage_channels[0], cfg.layers[0], strides[0], act, norm_fn
        )
        self.layer2 = self._make_layer(
            stage_channels[1], cfg.layers[1], strides[1], act, norm_fn
        )
        self.layer3 = self._make_layer(
            stage_channels[2], cfg.layers[2], strides[2], act, norm_fn
        )
        self.layer4 = self._make_layer(
            stage_channels[3], cfg.layers[3], strides[3], act, norm_fn
        )

        self.head_pool = _get_adaptive_head_pooling(cfg.head_pooling)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.fc = nn.Linear(stage_channels[3] * BasicBlock.expansion, cfg.num_classes)

        # 默认 Kaiming 初始化卷积；全连接用较小方差高斯（与常见 CIFAR 式实现一致）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride, activation, norm_layer):
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(
                BasicBlock(
                    self.in_planes,
                    planes,
                    stride=s,
                    activation=activation,
                    norm_layer=norm_layer,
                    use_bn=self.use_bn,
                )
            )
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """stem → layer1…4 → 全局池化 → flatten → dropout → logits。"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def build_resnet(cfg: SimpleNamespace) -> nn.Module:
    """根据 cfg.arch 构建模型；当前仅支持 `resnet18_manual`。"""
    if cfg.arch != "resnet18_manual":
        raise ValueError(f"unknown arch {cfg.arch}")
    if len(cfg.layers) != 4:
        raise ValueError("layers must define 4 stages")
    return ResNetManual(cfg)
