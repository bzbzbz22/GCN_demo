
import re
import argparse
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import os
import random
import dgl
from utils import load_data, accuracy
from gcn_dgl import GCN_dgl
from gcn import GCN

def parameter_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--use_dgl', action='store_true', default=False,
                    help='Using dgl to constract GCN. Default is False.')
    ap.add_argument('--dataset', type=str, default='texas')
    ap.add_argument('--hidden_dim', type=int, default=64)
    ap.add_argument('--num_layers', type=int, default=2)

    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--weight_decay', type=float, default=0.0005)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--no_cuda', action='store_false', default=True,
                    help='Using CUDA or not. Default is True (Using CUDA).')

    args, _ = ap.parse_known_args()
    args.device = torch.device('cuda' if args.no_cuda and torch.cuda.is_available() else 'cpu')

    return args


def setup_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    if cuda is True:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


args = parameter_parser()
print("Whether to use GCN built by DGL: {}".format(args.use_dgl))
setup_seed(args.seed, torch.cuda.is_available())

class_num = {'texas':5, 'wisconsin':5, 'cora':7}

for repeat in range(2):
    print('Repeat {}: '.format(repeat))

    for run in range(5):
        g, features, train_idx, val_idx, test_idx, labels = load_data(args.dataset, run, args.device)

        if args.use_dgl == True:
            model = GCN_dgl(nfeat=features.shape[1],
                            nhid=args.hidden_dim,
                            nclass=class_num[args.dataset],
                            nlayers=args.num_layers,
                            dropout=args.dropout)
        else:
            model = GCN(nfeat=features.shape[1],
                            nhid=args.hidden_dim,
                            nclass=class_num[args.dataset],
                            nlayers=args.num_layers,
                            dropout=args.dropout)

        model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):

            model.train()
            optimizer.zero_grad()
            logits = model(g, features)
            train_loss = F.nll_loss(torch.log(logits)[train_idx], labels[train_idx])
            train_acc = accuracy(logits[train_idx], labels[train_idx])
            train_loss.backward()
            optimizer.step()

        model.eval()
        logits = model(g, features)

        test_acc = accuracy(logits[test_idx], labels[test_idx])
        print('Run {}, test_acc: {:.4f}'.format(run, test_acc))

