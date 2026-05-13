# STL-10 手写 ResNet 可配置训练框架

本项目在 PyTorch 中从头实现 ResNet-18 风格的残差网络（不使用 `torchvision.models.resnet`），配合 YAML 配置完成数据增强、结构选项与优化器的快速对比实验。

## 数据布局

将 STL-10 按类别文件夹放在仓库根目录下的 `STL10/`：

- `STL10/train/`：7000 张（训练 + 从中划分验证）
- `STL10/test/`：1000 张（仅训练结束后评估一次）

训练与验证阶段**不得**读取 `test`。

若数据中存在 **0 字节或损坏图片**，`src/data/dataset.py` 中的 `FilteredImageFolder` 会跳过这些路径，分层划分与训练索引均基于过滤后的样本列表。

验证集默认从训练集中按类别 **分层随机划分 15%**（约每类 120 张验证、680 张训练），划分索引缓存在对应实验的 `artifacts/.../splits/` 下，便于复现。

## 环境

```bash
cd /data/home/dyq/programme_2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 常用命令

在项目根目录执行（确保当前目录为 `programme_2`，以便相对路径 `STL10` 与 `configs` 生效）：

```bash
python main.py train --config configs/default.yaml
python main.py eval_test --config configs/default.yaml
python main.py gradcam --config configs/default.yaml
```

使用 `nohup ... > train.log 2>&1 &` 时，标准输出被判定为非交互终端：会按 `logging.train_log_every_n_batches`（默认每个 batch 一行）写入带时间戳的 `loss` / `avg_loss` / `batch_acc` / `run_acc` / `lr` / `top-k`；本机交互终端仍使用 `tqdm` 动态刷新。

- `train`：训练并在 `artifacts/<run>/` 写入 `best.pt`、`metrics_history.json`、`curves_loss_acc.png`、`config_used.yaml`。默认 `logging.eval_on_test_after_train: true`，训练结束后会**自动**在测试集上跑一次测评（与单独执行 `eval_test` 相同产物）。默认 `gradcam.run_after_train: true`，会在其后自动生成 `gradcam_samples.png`。
- `eval_test`：加载 `best.pt`，在 **测试集** 上输出 `test_report.txt`、`test_confusion_matrix.png`、`test_topk_bar.png`、`test_per_class_recall_f1.png`。若已将 `eval_on_test_after_train` 设为 `false`，需在本训练结束后再手动运行此命令。
- `gradcam`：在测试集随机抽样图像上生成 `gradcam_samples.png`。若训练时已自动跑过，可省略；也可单独改参数后再执行以重新出图。

## 对比实验预设

`configs/experiments/` 下提供三组仅改关键项的完整配置（各自输出到不同 `artifacts_dir`）：

| 文件 | 侧重 |
|------|------|
| `exp_heavy_augmentation.yaml` | 更强增强 + Dropout |
| `exp_adam_cosine.yaml` | AdamW + 较小 eta_min 的余弦退火 |
| `exp_max_pool_head.yaml` | GELU + 全局最大池化分类头 |

示例：

```bash
python main.py train --config configs/experiments/exp_adam_cosine.yaml
```

## 源码结构（节选）

- `main.py`（项目根目录）：CLI（`train` / `eval_test` / `gradcam`）。
- `src/models.py`：手写 ResNet（`build_resnet`）。
- `src/train.py`：YAML 配置加载与路径解析（`load_config`、`resolve_paths`）、`Config` 类型别名、随机种子、优化器/调度器/损失、`train_epoch` / `evaluate`。
- `src/data/dataset.py`：`FilteredImageFolder`、类别名、训练/验证分层划分与划分缓存。
- `src/data/transforms.py`：YAML 列表 → `torchvision` 变换；`augmentation_enabled` 为 true 时训练默认保证至少含 **RandomHorizontalFlip** 与 **RandomCrop**（可在 YAML 中显式写，缺失时自动补全）。
- `src/metrics/`：指标与绘图。

## 可插拔配置说明（摘要）

运行时优先读取 [`configs/default.yaml`](configs/default.yaml) 作为默认值，再与命令行传入的 YAML **深度合并**（后者覆盖同名键）。合并结果在内存中为可点号访问的结构（见 `src/train.py` 中的 `load_config`）。

- **数据**：`data.augmentation_enabled` 为总开关（关则训练不做随机增强）；`data.augmentation_train` 由 `src/data/transforms.py` 映射；开启时若未写水平翻转或随机裁剪会自动补上（`ensure_min_train_aug=True`）。
- **模型**：`activation`、`use_bn`、`dropout`、`head_pooling` 等见 `configs/default.yaml`。
- **优化 / 调度**：`optimizer.name` 支持 `sgd` / `adam` / `adamw`；`scheduler.name` 支持 `cosine` / `step` / `multistep` / `none`（余弦退火默认 `T_max` 等于训练 epoch 数）。
- **Grad-CAM**：`gradcam.run_after_train` 控制训练结束后是否自动生成热力图（默认 true）。

## 额外指标与可视化

除 Precision / Recall / F1 与混淆矩阵外，实现：

1. **Top-k 准确率**（默认 k=3、5）：测试汇总写入报告并生成 `test_topk_bar.png`。
2. **各类 recall 与 F1 条形图**：`test_per_class_recall_f1.png`。

## 过拟合判断提示

对比 `curves_loss_acc.png` 中训练与验证曲线：若训练损失持续下降且训练准确率显著高于验证，而验证指标停滞或变差，则呈现典型过拟合迹象。

## AI 工具声明（作业用）

若在报告中需声明 AI 辅助，请根据自身使用情况填写工具名称与用途（例如：架构草稿、代码骨架、文档 wording 等）。
