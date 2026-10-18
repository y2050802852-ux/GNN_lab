<<<<<<< HEAD
16.6 项目目录规范(把 15 章落成仓库)

四个项目做完,是时候把散装脚本收进标准仓库结构——这是你开自己课题仓库的模板:s

```
gnn-project/
├── configs/                  # 一个实验一个 yaml(15.2)
│   ├── cora_gcn.yaml
│   └── mutag_gin.yaml
├── src/
│   ├── data.py               # 数据加载与预处理(含 CiteGraph 生成器)
│   ├── models/               # gcn.py / sage.py / gat.py / gin.py —— 手写版与 PyG 版并列
│   ├── engine.py             # Trainer(15.6)+ train/eval 函数
│   └── utils.py              # set_seed / setup_logger / dump_result
├── scripts/
│   └── train.py              # 唯一入口:python scripts/train.py --config configs/cora_gcn.yaml
├── runs/                     # 产物:checkpoint / 日志 / tb / result_*.json(全部进 .gitignore)
├── tests/                    # 手写层 vs PyG 层的 allclose 对拍(16.2 的验证自动化)
└── README.md
```

=======

# GNN_lab

一个简洁、专业、方便长期扩展的名字，不会把项目局限在某一种 GNN 模型或具体任务上。

>>>>>>> 1f9e188270209e0f1601e7175310d3e393bc5806
>>>>>>>
>>>>>>
>>>>>
>>>>
>>>
>>
