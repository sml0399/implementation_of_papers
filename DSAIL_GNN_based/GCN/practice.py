import torch
import torch_geometric 
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(dataset[0].edge_index[0].shape)

