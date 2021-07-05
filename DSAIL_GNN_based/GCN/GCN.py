import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, channel_in, channel_middle, channel_out, do_normalization, use_bias):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=channel_in, out_channels=channel_middle, normalized=do_normalization, bias=use_bias)
        self.conv2 = GCNConv(in_channels=channel_middle, out_channels=channel_out, normalized=do_normalization, bias=use_bias)
    def forward(self, x):
        
        return
