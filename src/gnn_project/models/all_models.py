import torch,torch_geometric
import torch.nn as nn
from torch_geometric import nn as gnn
import yaml
from torch_geometric.nn import global_add_pool

class GCN(nn.Module):
    def __init__(self,in_dim,n_classes,hidden=64, dropout=0.5,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = gnn.GCNConv(in_dim,hidden)
        self.conv2 = gnn.GCNConv(hidden,n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,edge_index):
        h = self.conv1(x,edge_index).relu()
        h = self.dropout(h)
        return self.conv2(h,edge_index)


class SAGE(nn.Module):
    def __init__(self,in_dim,n_classes,hidden=64,dropout=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = gnn.SAGEConv(in_dim,hidden)
        self.conv2 = gnn.SAGEConv(hidden,n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,edge_index):
        h = self.conv1(x,edge_index).relu()
        h = self.dropout(h)
        return self.conv2(h,edge_index)


class GIN(nn.Module):
    def __init__(self,in_dim,n_classes,hidden=64,dropout=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mlp1 = nn.Sequential(
            nn.Linear(in_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden)
        )
        mlp2 = nn.Sequential(
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden)
        )
        self.conv1 = gnn.GINConv(mlp1)
        self.conv2 = gnn.GINConv(mlp2)
        self.dropout = nn.Dropout(dropout)
        self.classfier = nn.Linear(hidden,n_classes)

    def forward(self,x,edge_index,batch):
        h = self.conv1(x,edge_index).relu()
        h = self.dropout(h)
        h = self.conv2(h,edge_index).relu()
        h = global_add_pool(h,batch)
        logits = self.classfier(h)
        return logits


class LinkPredGCN(nn.Module):
    def __init__(self,in_dim,n_classes,hidden=64,dropout=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = gnn.GCNConv(in_dim,hidden)
        self.conv2 = gnn.GCNConv(hidden,hidden)
        self.dropout = nn.Dropout(dropout)
        self.decode_mlp = nn.Sequential(
            nn.Linear(3*hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,1)
        )

    def forward(self,x,edge_index):
        h = self.conv1(x,edge_index).relu()
        h = self.dropout(h)
        return self.conv2(h,edge_index)


    def decode(self,z,edge_label_index):    # 对候选边打分
        src,dst = edge_label_index
        return self.decode_mlp(torch.cat([z[src],z[dst], z[src]*z[dst]],dim=-1)).squeeze(-1)
        # return (z[src] * z[dst]).sum(-1)    # 内积


def get_model(model_name,in_dim,n_classes,hidden,dropout,task,model_path=None):
    config_path = './configs/model_registry.yaml'
    with open(config_path, 'r',encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    model_name = model_name.lower()
    task=task.lower()

    models = model_config['models']
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    config = models[model_name]
    if task not in config['tasks']:
        raise ValueError(f"Model {model_name} does not support task {task}"
                         f"Supported tasks: {config['tasks']}")

    model_classes = {
        "GCN": GCN,
        "SAGE": SAGE,
        "GIN": GIN,
        "LinkPredGCN": LinkPredGCN,
    }

    class_name = config["class"]
    model_cls = model_classes[class_name]
    if model_path is not None:
        model =  model_cls(in_dim, n_classes, hidden, dropout)
        checkpoint = torch.load(
            model_path,
            map_location='cpu',
            weights_only=True
        )
        model.load_state_dict(checkpoint['model'])
        return model
    # 5. 创建并返回模型
    return model_cls(in_dim, n_classes, hidden, dropout)

