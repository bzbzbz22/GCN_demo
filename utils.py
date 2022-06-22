import os
import random
import dgl
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F


def load_data(dataset, repeat, device):
    path = './data_geom/{}/'.format(dataset)

    struct_edges = np.genfromtxt(path + '{}.edge'.format(dataset), dtype=np.int32)

    U = [e[0] for e in struct_edges]
    V = [e[1] for e in struct_edges]
    g = dgl.graph((U, V))
    g = dgl.to_simple(g)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = dgl.to_bidirected(g)

    g = g.to(device)
    deg = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    norm[torch.isinf(norm)] = 0
    norm = norm.to(device)
    # g.ndata['d'] = norm.unsqueeze(1)
    g.ndata['d'] = norm


    f = np.loadtxt(path + '{}.feature'.format(dataset), dtype=float)
    l = np.loadtxt(path + '{}.label'.format(dataset), dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense())).to(device)
    label = torch.LongTensor(np.array(l)).to(device)


    test = np.loadtxt(path + '{}test.txt'.format(repeat), dtype=int)
    train = np.loadtxt(path + '{}train.txt'.format(repeat), dtype=int)
    val = np.loadtxt(path + '{}val.txt'.format(repeat), dtype=int)

    idx_test = test.tolist()
    idx_train = train.tolist()
    idx_val = val.tolist()

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)

    return g, features, idx_train, idx_val, idx_test, label


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)