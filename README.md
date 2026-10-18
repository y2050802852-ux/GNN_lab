# GNN Lab

一个基于 PyTorch Geometric 的图神经网络实验项目，提供节点分类、图分类和链路预测的统一训练流程。项目使用 YAML 配置实验参数，并支持自动选择 CUDA、Apple MPS 或 CPU 设备。

> 当前项目更适合作为 GNN 学习和实验模板。不同任务的代码路径已经分开，但部分模型和数据集组合仍属于实验性支持，详见[已知限制](#已知限制)。

## 功能概览

- 节点分类：Cora、CiteSeer、PubMed
- 图分类：OGB `ogbg-molhiv`，以及 TUDataset 中的 PROTEINS、MUTAG、ENZYMES
- 链路预测：Cora，通过 `RandomLinkSplit` 生成训练、验证和测试边
- 模型：GCN、GraphSAGE、GIN、LinkPredGCN
- 训练功能：
  - Adam 优化器
  - StepLR 学习率调度
  - Early stopping
  - 最优 checkpoint 保存
  - CSV 指标保存
  - TensorBoard 训练曲线
  - 随机种子初始化
  - CUDA/MPS/CPU 自动设备选择

## 项目结构

```text
GNN_lab/
├── configs/
│   ├── config.yaml          # 当前实验配置
│   └── model_registry.yaml  # 模型与任务注册表
├── data/                    # 本地数据集，运行时下载或处理，默认被 Git 忽略
├── runs/                    # checkpoint、日志、CSV 和 TensorBoard 输出，默认被 Git 忽略
├── scripts/
│   ├── train.py             # 训练入口
│   └── predict.py           # 加载 checkpoint 并生成预测
├── src/gnn_project/
│   ├── data.py              # 数据集加载、划分和 DataLoader
│   ├── engine.py            # Trainer、训练/验证/测试和预测逻辑
│   ├── utils.py             # 配置、日志、随机种子、设备和 CSV 工具
│   └── models/
│       └── all_models.py    # GCN、SAGE、GIN、LinkPredGCN
├── dataset_autodownload.py  # 下载并检查 ogbg-molhiv 的辅助脚本
├── requirements.txt         # Python 依赖
├── pyproject.toml           # 本地 Python 包配置
└── README.md
```

## 环境安装

建议使用 Python 3.11 或更高版本。当前 `requirements.txt` 的版本是在 Apple Silicon Mac 的本地环境中验证的，包含 PyTorch、PyTorch Geometric、OGB、NumPy、Pandas、SciPy、scikit-learn、PyYAML 和 TensorBoard。

在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果使用 Conda，也可以直接在已有环境中执行：

```bash
python -m pip install -r requirements.txt
```

### PyTorch 和设备说明

`torch` 和 `torch-geometric` 对 Python、操作系统和 CUDA 版本比较敏感。如果使用 NVIDIA GPU，建议按照自己的 CUDA 版本先安装对应的 PyTorch，再安装与其兼容的 PyG 版本。Apple Silicon 用户可以使用 MPS；项目会自动按以下优先级选择设备：

```text
CUDA > Apple MPS > CPU
```

## 运行前提

请始终从项目根目录运行命令，因为当前脚本使用了相对路径：

```text
./configs/config.yaml
./configs/model_registry.yaml
./data/
```

例如：

```bash
cd /path/to/GNN_lab
python scripts/train.py
```

## 支持的任务、数据集和模型

### 节点分类

| 数据集配置名 | PyG 数据集 | 可用模型 |
|---|---|---|
| `cora` | `Planetoid("Cora")` | `gcn`、`sage` |
| `citeseer` | `Planetoid("CiteSeer")` | `gcn`、`sage` |
| `pubmed` | `Planetoid("PubMed")` | `gcn`、`sage` |

节点分类使用数据集提供的 `train_mask`、`val_mask` 和 `test_mask`。GraphSAGE 使用 `NeighborLoader` 做邻居采样。

### 图分类

| 数据集配置名 | 数据集 | 可用模型 |
|---|---|---|
| `obg` 或 `ogbg-molhiv` | OGB `ogbg-molhiv` | `gin` |
| `proteins` | TUDataset `PROTEINS` | `gin` |
| `mutag` | TUDataset `MUTAG` | `gin` |
| `enzymes` | TUDataset `ENZYMES` | `gin` |

OGB-MolHIV 使用官方 train/valid/test 划分，训练损失使用二元交叉熵，验证指标为 ROC-AUC。

### 链路预测

当前支持：

```yaml
task: edge
dataset: cora
model: linkpredgcn
```

Cora 的真实引用边作为正样本，`RandomLinkSplit` 生成负样本并划分训练、验证和测试数据。LinkPredGCN 使用 GCN 编码节点表示，再通过边解码器给候选边打分，训练损失为 `BCEWithLogitsLoss`。

## 配置实验

主要配置文件是 `configs/config.yaml`。一个链路预测配置示例：

```yaml
task: edge
model: linkpredgcn
dataset: cora
hidden: 128
dropout: 0.5
epochs: 300
seed: 42
earlystop: true
patience: 50

sched:
  enable: true
  step_size: 20
  gamma: 0.5

lr: 0.001
weight_decay: 5e-4

log:
  out_dir: runs/GNN_out_log
  tensorboard: true

randomseed: 0
npseed: 0
torchseed: 0
batch_size: 128
num_neighbors: [25, 15]
```

常用字段：

- `task`：`node`、`graph` 或 `edge`
- `model`：模型注册名
- `dataset`：数据集配置名
- `hidden`：隐藏层维度
- `dropout`：Dropout 概率
- `epochs`：最大训练轮数
- `earlystop`、`patience`：是否启用早停及容忍轮数
- `sched.enable`、`sched.step_size`、`sched.gamma`：StepLR 设置
- `lr`、`weight_decay`：Adam 参数
- `log.out_dir`：实验输出根目录
- `batch_size`：图分类 batch 大小或 GraphSAGE 的采样 batch 大小
- `num_neighbors`：GraphSAGE 每层采样邻居数量

当前随机数初始化实际使用的是 `randomseed`、`npseed` 和 `torchseed`。配置中的 `seed` 主要用于 TensorBoard run 名称和实验标识。

## 训练

### 使用 YAML 配置

```bash
python scripts/train.py
```

训练入口会读取：

```text
configs/config.yaml
```

### 使用命令行覆盖部分字段

当前脚本支持覆盖以下顶层字段：`model`、`hidden`、`dropout`、`epochs`、`lr`、`seed`。

```bash
python scripts/train.py \
  --model sage \
  --hidden 256 \
  --dropout 0.5 \
  --epochs 300 \
  --lr 0.001 \
  --seed 42
```

`task`、`dataset` 和嵌套配置建议直接修改 `configs/config.yaml`。链路预测使用 `linkpredgcn` 时，也建议直接在 YAML 中设置，因为当前命令行模型选项没有列出该模型。

## 数据集下载

第一次使用 PyG 数据集时，相关数据会自动下载到 `data/`。OGB-MolHIV 也可以使用辅助脚本预先下载：

```bash
python dataset_autodownload.py
```

该脚本会将数据保存到：

```text
./data/ogbg_molhiv/
```

数据文件通常较大，项目默认通过 `.gitignore` 忽略 `data/`，不会被提交到 GitHub。

## 预测

先在 `configs/config.yaml` 中设置一个实际存在的 checkpoint 路径，例如：

```yaml
load_model_path: /path/to/your/run/model_best.pt
```

然后从项目根目录执行：

```bash
python scripts/predict.py
```

预测结果会保存到本次预测输出目录下的：

```text
runs/GNN_out_log/Predict/<run_name>/Predict/predictions.pt
```

注意：配置中的 checkpoint 路径不能直接复用其他电脑上的绝对路径，需要根据本机文件位置修改。

## 输出文件

训练结果默认位于：

```text
runs/GNN_out_log/Train/
└── Training_Result_<task>_<model>_<dataset>_<timestamp>/
    ├── train_log
    ├── <model>_best.pt
    ├── metrics_loss.csv
    └── ...
```

TensorBoard 日志单独写入：

```text
runs/GNN_out_log/tb/<model>-<hidden>-lr<lr>-seed<seed>/
```

checkpoint 包含：

- 模型参数
- 优化器状态
- 学习率调度器状态
- 当前 epoch
- 最优验证指标
- 当前配置
- Early stopping 状态

TensorBoard 会记录：

- `loss/train`
- `acc/val`
- `lr`
- `grad/norm`

启动 TensorBoard：

```bash
tensorboard --logdir runs/GNN_out_log/tb
```

## 已知限制

当前版本仍有一些实验性限制，使用时请注意：

1. 脚本依赖从项目根目录执行；从其他目录启动可能找不到配置文件或数据目录。
2. 当前 `config.yaml` 中的 `load_model_path` 可能是某台机器上的绝对路径，预测前必须替换为本机 checkpoint 路径。
3. `optimizer_name` 字段目前没有参与优化器构造，训练代码固定使用 Adam。
4. `data.homophily` 和 `data.train_per_class` 只对预留的合成图生成器有意义，不是当前 Planetoid/OGB/TUDataset 流程的通用配置。
5. `evaluate_graph_epoch` 当前使用 OGB-MolHIV 的 `Evaluator`，因此其他 TUDataset 图分类数据集的评估流程仍需要进一步适配。
6. CLI 参数列表中仍保留了 `gat`，但当前模型注册表和模型实现没有 GAT；`linkpredgcn` 已注册，但建议通过 YAML 选择。
7. `tests/` 当前没有完整的自动化测试，README 中的支持范围表示已有代码路径，不代表所有模型与任务组合都经过完整基准验证。
8. `data/`、`runs/`、模型 checkpoint 和 TensorBoard 事件文件默认被 Git 忽略，因此不会随着代码仓库上传。

## 许可证

当前仓库未声明正式开源许可证。如需允许其他人明确地复制、修改和分发代码，建议后续添加合适的 `LICENSE` 文件。
