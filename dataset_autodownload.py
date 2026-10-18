
from ogb.graphproppred import PygGraphPropPredDataset

# 下载并处理数据集，保存到 ./data/ogbg_molhiv/
dataset = PygGraphPropPredDataset(
    name="ogbg-molhiv",
    root="./data"
)

print("数据集下载完成！")
print("图数量：", len(dataset))
print("类别数量：", dataset.num_classes)
print("第一张图：", dataset[0])
print("第一张图的节点特征形状：", dataset[0].x.shape)
print("第一张图的边索引形状：", dataset[0].edge_index.shape)
