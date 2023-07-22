import os
import argparse
import time

import random
import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F

from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import networkx as nx


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

    dataset = PygNodePropPredDataset(name='ogbn-products')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    ## Convert split indices to boolean masks and add them to `data`.
    #for key, idx in split_idx.items():
    #    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    #    mask[idx] = True
    #    data[f'{key}_mask'] = mask

    E = data.edge_index.T.tolist()
    G = nx.Graph(E)
    G.add_nodes_from(list(range(data.x.shape[0])))
    print(type(G))
    print(G.number_of_nodes(), G.size())
    start = time.time()
    pr = nx.pagerank(G)
    end = time.time()
    print(time.localtime(start), time.localtime(end))
    torch.save(pr, os.path.join('age', 'pr.pt'))


if __name__ == "__main__":
    main()
