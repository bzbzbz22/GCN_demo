import math

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
# from dgl import function as fn
from dgl.nn.pytorch import GraphConv


class GCN_dgl(nn.Module):
    def __init__(self,
                 nfeat,
                 nlayers,
                 nhid,
                 nclass,
                 dropout):
        super(GCN_dgl, self).__init__()
        # self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(nfeat, nhid))
        for i in range(nlayers - 1):
            self.layers.append(GraphConv(nhid, nhid))
        self.layers.append(GraphConv(nhid, nclass))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, x):
        h = x
        for i, layer in enumerate(self.layers):
            if i!=0:
                h = self.dropout(h)
            h = layer(g, h)
        # return h
        return F.softmax(h, 1)