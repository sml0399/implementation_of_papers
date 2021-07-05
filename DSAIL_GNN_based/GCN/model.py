import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg # pytorch geometric

class GCN(torch.nn.Module):
    def __init__(self, channel_in, channel_middle, channel_out, do_normalization, use_bias):
        super(GCN, self).__init__()
        self.conv1 = pyg.GCNConv(in_channels=channel_in, out_channels=channel_middle, normalized=do_normalization, bias=use_bias)
        self.conv2 = pyg.GCNConv(in_channels=channel_middle, out_channels=channel_out, normalized=do_normalization, bias=use_bias)

    def forward(self, init_node_features, edge_index):
        '''
        Description:
            forward operation
        
        Input:
            init_node_features: initial node features. Tensor with shape [number_of_nodes, number_of_features_per_node]
            edge_index: edge list. Tensor with shape [2, number_of_edges]
        '''
        x = self.conv1(init_node_features, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
