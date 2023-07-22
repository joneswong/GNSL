import os
import argparse
import time

import random
import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch_sparse import SparseTensor

import torch_geometric.transforms as T
from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree, subgraph

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import networkx as nx
import walker


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (GraphSAINT)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--inductive', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--walk_length', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_steps', type=int, default=2)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    adj_t = data.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-1.0)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t
    print(adj_t)
    v = torch.zeros(data.num_nodes)
    v[3] = 1.0
    rs = torch.mv(torch.mv(adj_t, torch.mv(adj_t, v)))
    print(rs)

    """
    dataset = PygNodePropPredDataset(name='ogbn-products')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    E = data.edge_index.T.tolist()
    G = nx.Graph(E)
    G.add_nodes_from(list(range(data.x.shape[0])))
    print(type(G))
    print(G.number_of_nodes(), G.size())
    start = time.time()
    X = walker.random_walks(G, n_walks=20, walk_len=3, start_nodes=[0, 1, 2])
    print(X)
    print(time.localtime(start), time.localtime(end))
    """


if __name__ == "__main__":
    main()
