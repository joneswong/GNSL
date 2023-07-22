import os
import argparse

import random
import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F

from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from utils import *


def main():
    parser = argparse.ArgumentParser(description='Calc Random')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    print(args)

    setup_seed(123)

    dataset = PygNodePropPredDataset(name='ogbn-products')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    indices = split_idx['train'].cpu().numpy().tolist()
    vals = np.random.randn(len(indices)).tolist()
    results = [(indices[i], vals[indices[i]]) for i in range(len(indices))]
    results = sorted(results, key=lambda x:x[1], reverse=True)
    with open(os.path.join("random", "train.tsv"), 'w') as ops:
        for i in range(len(results)):
            idx, val = results[i]
            ops.write("{}\t{}\n".format(idx, val))


if __name__ == "__main__":
    main()
