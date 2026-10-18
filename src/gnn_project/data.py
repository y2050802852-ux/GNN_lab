# 数据加载与预处理（含 CiteGraph 生成器等）
import os,yaml
import torch
from torch_geometric.datasets import Planetoid, TUDataset
from gnn_project.utils import load_config
from torch_geometric.loader import NeighborLoader,DataLoader
from torch_geometric.data import Data

def make_citegraph(n_nodes=400, n_classes=7, dim=16, homophily=0.6,
                   train_per_class=20, n_val=100, n_test=200, seed=42):
    """ CiteGraph 精简版(第 9.5 节;本章只用小图,让示例跑得快)。
        创建图函数
    """
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, dim, generator=g) * 2.0
    y = torch.randint(0, n_classes, (n_nodes,), generator=g)
    X = centers[y] + torch.randn(n_nodes, dim, generator=g) * 0.5
    same = torch.rand(n_nodes, generator=g) < homophily
    epn = 6
    src = torch.empty(n_nodes * epn, dtype=torch.long)
    dst = torch.arange(n_nodes).repeat_interleave(epn)
    for i in range(n_nodes):
        pool = ((y == y[i]) if same[i] else (y != y[i])).nonzero().squeeze(1)
        pick = pool[torch.randint(0, pool.size(0), (epn,), generator=g)]
        src[i * epn:(i + 1) * epn] = pick
    edges = torch.stack([src, dst], dim=0)
    edges = torch.cat([edges, edges.flip(0)], dim=1)
    loops = torch.arange(n_nodes)
    edges = torch.cat([edges, torch.stack([loops, loops])], dim=1)
    key = edges[0] * n_nodes + edges[1]
    key = torch.unique(key)
    edges = torch.stack([key // n_nodes, key % n_nodes], dim=0)
    perm = torch.randperm(n_nodes, generator=g)
    train_idx = torch.cat([(y == c).nonzero().squeeze(1)[
        torch.randperm((y == c).sum(), generator=g)[:train_per_class]]
        for c in range(n_classes)])
    rest = torch.tensor([i for i in perm.tolist() if i not in set(train_idx.tolist())])
    val_idx, test_idx = rest[:n_val], rest[n_val:n_val + n_test]
    data = Data(
    x=X,
    y=y,
    edge_index=edges,
    train_mask=torch.zeros(n_nodes, dtype=torch.bool),
    val_mask=torch.zeros(n_nodes, dtype=torch.bool),
    test_mask=torch.zeros(n_nodes, dtype=torch.bool),
    )

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    data.homophily = homophily
    # out = {'X': X, 'y': y, 'edge_index': edges, 'homophily': homophily}
    # for name, idx in [('train_mask', train_idx), ('val_mask', val_idx), ('test_mask', test_idx)]:
    #     m = torch.zeros(n_nodes, dtype=torch.bool); m[idx] = True
    #     out[name] = m

    return data


# class Dataset():
#     def __init__(self,name,path,cfg):
#         self.dataset = self.load_dataset(root=path,name=name,task=cfg['task'])
#         self.data = self.dataset[0]
#         if cfg['model'] == 'sage':
#             num_neighbors = cfg['num_neighbors']
#             self.loader = NeighborLoader(self.data,num_neighbors=num_neighbors,batch_size=256,
#                                     input_nodes=self.dataset.train_mask,shuffle=True)
            

#     @staticmethod
#     def load_dataset(name, task, root='./data'):
#         if name in {'Cora', "CiteSeer", "PubMed"} and task == 'node':
#             return Planetoid(root=root,name=name)
#         # elif name in {"PROTEINS", "MUTAG", "ENZYMES"}:
#         #     return TUDataset(root=root,name=name)
#         elif task == 'graph' and name == 'obg':
#             from ogb.graphproppred import PygGraphPropPredDataset
#             return PygGraphPropPredDataset(
#                             name="ogbg-molhiv",
#                             root="./data"
#                             )
        
#         raise ValueError(f'Unsupported datasets: {name} and {task}')

class Dataset():
    def __init__(self, name, path, cfg):
        self.name = name.lower()
        self.task = cfg['task'].lower()
        self.model = cfg['model'].lower()
        self.root = path

        self.dataset = self.load_dataset(
            name=name,
            task=self.task,
            root=self.root
        )

        self.data = None
        self.split_idx = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.in_dim = None
        self.n_classes = None
        if self.task == 'node':
            self._setup_node_task(cfg)
            self.data = self.dataset.data
            self.in_dim = self.data.x.size(1)
            self.n_classes = self.dataset.num_classes

        elif self.task == 'graph':
            self._setup_graph_task(cfg)
            self.data = self.dataset[0]
            self.in_dim = self.data.x.size(1)

            if cfg['dataset'].lower() in {'obg', 'ogbg-molhiv'}:
                self.n_classes = self.dataset.num_tasks
            else:
                self.n_classes = self.dataset.num_classes

        elif self.task == 'edge':
            self._setup_edge_task(cfg)
            self.data = self.train_data
            self.in_dim = self.data.x.size(1)
            self.n_classes = 2
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    @staticmethod
    def load_dataset(name, task, root='./data'):
        """根据数据集名称和任务加载数据集"""
        # 节点分类数据集
        if task == 'node':
            node_datasets = {
                'cora': "Cora",
                'citeseer': "CiteSeer",
                'pubmed': "PubMed",
            }
            if name in node_datasets:
                return Planetoid(
                    root,
                    name=node_datasets[name]
                )
        # 图分类数据集
        elif task == 'graph':
            if name in {'obg', 'obg-molhiv'}:
                from ogb.graphproppred import PygGraphPropPredDataset

                return PygGraphPropPredDataset(
                    name='ogbg-molhiv',
                    root=root,
                )

            graph_datasets = {
                'proteins': "PROTEINS",
                "mutag": "MUTAG",
                'enzymes': "ENZYMES",
            }

            if name in graph_datasets:
                return TUDataset(
                    root=root,
                    name=graph_datasets[name]
                )
        elif task == 'edge':
            if name == 'cora':
                from torch_geometric.transforms import ToUndirected

                return Planetoid(
                    root=root,
                    name='Cora',
                    transform=ToUndirected(),
                )
        raise ValueError(
            f'Unsupported dataset: {name} for task {task}'
        )

    def _setup_node_task(self,cfg):
        """节点分类：整图训练或者邻居采样"""
        self.data = self.dataset[0]

        if self.model == 'sage':
            num_neighbors = cfg.get('num_neighbors', [10,10])
            batch_size = cfg.get('batch_size', 256)

            self.train_loader = NeighborLoader(
                self.data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=self.data.train_mask,
                shuffle=True
            )

            self.valid_loader = NeighborLoader(
                self.data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=self.data.val_mask,
                shuffle=True
            )
            
            self.test_loader = NeighborLoader(
                self.data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=self.data.test_mask,
                shuffle=True
            )

    def _setup_graph_task(self,cfg):
        """图分类: 划分图样本并创建 Dataloader"""
        batch_size = cfg.get('batch_size', 256)

        # OGB 使用官方划分
        if self.name in {'obg', 'ogbg-molhiv'}:
            self.split_idx = self.dataset.get_idx_split()

            train_dataset = self.dataset[self.split_idx['train']]
            valid_dataset = self.dataset[self.split_idx['valid']]
            test_dataset = self.dataset[self.split_idx['test']]

        else:
            n = len(self.dataset)
            generator = torch.Generator().manual_seed(
                cfg.get('torchseed', 42)
            )
            indices = torch.randperm(n,generator=generator)

            n_train = int(n * 0.8)
            n_valid = int(n * 0.1)

            train_idx = indices[:n_train]
            valid_idx = indices[n_train:n_train + n_valid]
            test_idx = indices[n_train + n_valid:]

            self.split_idx = {
                'train': train_idx,
                'valid': valid_idx,
                'test': test_idx,
            }

            train_dataset = self.dataset[train_idx.tolist()]
            valid_dataset = self.dataset[valid_idx.tolist()]
            test_dataset = self.dataset[test_idx.tolist()]

        self.train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        )
        self.valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        )
        self.test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        ) 
    def _setup_edge_task(self,cfg):
        """边预测： 划分正负边，并准备训练、验证和测试数据"""

        from torch_geometric.transforms import RandomLinkSplit

        if cfg['dataset'] == 'cora':
            data = self.dataset[0]

            splitter = RandomLinkSplit(
                num_val=0.05,
                num_test=0.1,
                is_undirected=True,
                add_negative_train_samples=True,
                neg_sampling_ratio=1.,
                split_labels=False,
            )
            self.train_data, self.valid_data, self.test_data = splitter(data)

            self.data = self.train_data

            self.n_classes = 2



    