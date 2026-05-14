# STL-10 手写 ResNet 可配置训练

在 **PyTorch** 中从零实现 **ResNet-18 风格**残差网络（不调用 `torchvision.models.resnet`），用 **YAML** 统一配置数据增强、网络选项与优化器，便于做对照实验。

---

## 目录

1. [功能概览](#功能概览)  
2. [数据准备](#数据准备)  
3. [环境与运行](#环境与运行)  
4. [命令说明](#命令说明)  
5. [配置怎么生效](#配置怎么生效)  
6. [输出文件](#输出文件)  
7. [代码结构](#代码结构)  
8. [指标与可视化](#指标与可视化)  
9. [过拟合怎么粗看](#过拟合怎么粗看)

---

## 功能概览

| 能力 | 说明 |
|------|------|
| 训练 / 验证 | 从 `train/` 分层划分验证集，**训练阶段不读 `test/`** |
| 测试集评估 | `eval_test`：分类报告、混淆矩阵、Top-k、每类 recall/F1 |
| 可解释性 | **Grad-CAM**（非经典 CAM），默认训练结束后可自动生成热力图 |

---

## 数据准备

在**项目根目录**（与 `main.py` 同级）放置 **STL-10**，按 **ImageFolder** 布局：

```text
STL10/
  train/          # 约 7000 张，每类一个子文件夹
    airplane/
    bird/
    ...
  test/           # 约 1000 张，结构同 train
    ...
```

- **训练与验证**只使用 `STL10/train/`；**测试**仅在 `eval_test` 或训练结束自动测评时使用 `STL10/test/`。  
- 若存在 **0 字节或损坏图片**，`FilteredImageFolder` 会过滤掉，划分与类别名都基于过滤后的样本。  
- 验证集默认从训练集 **按类分层随机划分** 一部分比例，具体数值见 `configs/default.yaml` 中的 `repro.val_ratio`。划分索引可缓存在本次运行的 `artifacts/.../splits/`，便于复现。

---

## 环境与运行

在仓库根目录执行（保证相对路径 `STL10/`、`configs/` 正确）：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py train --config configs/default.yaml
```

---

## 命令说明

| 子命令 | 作用 |
|--------|------|
| `train` | 训练；写入 `artifacts/<本次配置里的 artifacts_dir>/` 下的权重、曲线、历史 JSON，并复制 `config_used.yaml`。若配置中开启，会在结束后自动跑测试评估与 Grad-CAM。 |
| `eval_test` | 加载同目录下的 `best.pt`，在 **测试集** 上生成报告与图表（训练未完成或关闭自动测评时可单独执行）。 |
| `gradcam` | 在测试集上随机抽样，生成 `gradcam_samples.png`（三列：原图预览、热力图、叠加图 + 预测类名）。 |

**日志行为**：交互终端下训练循环使用 `tqdm`；若标准错误**不是** TTY（例如 `nohup ... > train.log 2>&1 &`），则按 `logging.train_log_every_n_batches` 定期打印带时间戳的 loss / acc / lr / top-k 等行，便于落盘排查。

**对比多组实验**：复制 `configs/default.yaml`（或另存为 `configs/my_exp.yaml`），只改 `paths.artifacts_dir` 等关键字段，再 `train --config configs/my_exp.yaml`，各次输出互不覆盖。

---

## 配置怎么生效

1. **加载顺序**：先读 `configs/default.yaml`，再与命令行 `--config` 指定的 YAML **深度合并**（后者覆盖同名键）。  
2. **项目根**：配置解析会从你传入的 YAML 路径向上查找含 `configs/default.yaml` 的目录作为根路径（一般即克隆下来的仓库根）。  
3. **常用键**（完整说明与默认值以 `configs/default.yaml` 内注释为准）：

| 区域 | 含义摘要 |
|------|----------|
| `paths` | `data_root`（如 `STL10`）、`artifacts_dir`（输出子目录） |
| `repro` | 随机种子、验证比例、是否缓存划分 |
| `data` | batch、增强列表、`augmentation_enabled` 总开关 |
| `model` | `activation`、`use_bn`、`dropout`、`head_pooling`、`stem_maxpool` 等 |
| `optimizer` / `scheduler` | 名称与小写枚举见 YAML 注释 |
| `logging` | 最佳权重依据、`eval_on_test_after_train`、batch 日志间隔 |
| `gradcam` | 目标层名、样本数、叠加透明度 `alpha` |

训练阶段若开启增强且未显式写出水平翻转或随机裁剪，会在代码侧 **自动补** 最低限度增强（见 `src/data/transforms.py`）。

---

## 输出文件

在 `paths.artifacts_dir` 所指目录下，常见文件包括：

| 文件 | 说明 |
|------|------|
| `best.pt` | 验证集上最优 checkpoint（含 `model_state`、`classes` 等） |
| `config_used.yaml` | 本次训练使用的配置副本 |
| `metrics_history.json` | 每 epoch 的 train/val 指标 |
| `curves_loss_acc.png` | 损失与 Top-1 准确率曲线 |
| `test_report.txt` | 测试 loss、Top-k 与 `classification_report` 文本 |
| `test_confusion_matrix.png` 等 | 混淆矩阵、Top-k 柱状图、每类 recall/F1 |
| `gradcam_samples.png` | Grad-CAM 网格图 |
| `splits/*.json` | 训练/验证索引与划分元数据（若开启缓存） |

---

## 代码结构

| 路径 | 职责 |
|------|------|
| `main.py` | CLI：`train` / `eval_test` / `gradcam`；组 DataLoader、调训练与测评流程 |
| `src/train.py` | 配置合并与路径解析、随机种子、优化器/调度器/损失、单轮训练与 `evaluate` |
| `src/models.py` | `BasicBlock`、`ResNetManual`、`build_resnet` |
| `src/data/dataset.py` | `FilteredImageFolder`、分层划分与划分缓存 |
| `src/data/transforms.py` | YAML 增强列表 → `torchvision` 的 `Compose` |
| `src/metrics/` | Top-k、预测收集、绘图与 Grad-CAM |

---

## 指标与可视化

- **Precision / Recall / F1**：由 `sklearn.metrics.classification_report` 生成；在各类样本数相同时，**macro avg** 的 recall 与 **Top-1 accuracy** 数值一致，可对照 `test_report.txt` 理解。  
- **Top-k 准确率**（默认 k=3、5）：真实类出现在 logits Top-k 中的比例。  
- **Grad-CAM**：对目标 logit 回传梯度，在指定卷积层上生成显著性图（实现见 `src/metrics/gradcam.py`）。

---
