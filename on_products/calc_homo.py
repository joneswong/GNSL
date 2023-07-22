import os
import argparse

import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph
import torch_scatter

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

from utils import *


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=20000)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    # degree
    degrees = pyg.utils.degree(data.edge_index[0], data.num_nodes)

    # same label edges
    edge_indices = (data.y[data.edge_index[0]] == data.y[data.edge_index[1]]).nonzero().squeeze(1)
    edge_mask = torch.zeros(data.num_edges, dtype=torch.bool)
    edge_mask[edge_indices] = True

    print("Homophilic levels:")
    # overall homophilic level
    print("Graph: {}".format(len(edge_indices) / degrees.sum().item()))

    # homophilic level of train, val, and test set
    for key in split_idx.keys():
        spl_degrees = degrees[data[f'{key}_mask']]
        spl_edge_mask = torch.logical_and(edge_mask, data[f'{key}_mask'][data.edge_index[0]])
        spl_edge_indices = spl_edge_mask.nonzero().squeeze(1)
        print("{}: {}".format(key, len(spl_edge_indices) / spl_degrees.sum().item()))

        if key == 'train':
            out = torch.zeros(data.num_nodes, dtype=torch.long)
            torch_scatter.scatter(edge_mask.long(), data.edge_index[0], out=out)
            homo_levels = out / (degrees + 1e-6)
            metric = 1.0 - homo_levels
            results = [(i, float(metric[i].item())) for i in range(len(data['train_mask'])) if data['train_mask'][i] == True]
            results = sorted(results, key=lambda x:x[1], reverse=True)
            with open(os.path.join("homo-level", "train.tsv"), 'w') as ops:
                for i in range(len(results)):
                    idx, hl = results[i]
                    ops.write("{}\t{}\n".format(idx, hl))

    # homophilic levels of specific node(s)
    #homo_of_hard = []
    #for i in tqdm(range(0, len(hard_sample), 32)):
    #    node_batch = hard_sample[i:min(i+32, len(hard_sample))]
    #    if len(node_batch) < 32:
    #        break
    #    idx = (data.edge_index[0].unsqueeze(1) == torch.Tensor(node_batch).long().unsqueeze(0)).nonzero()
    #    idx_ = idx[:,0]
    #    src_edge_batch, tgt_edge_batch = data.edge_index[0][idx_], data.edge_index[1][idx_]
    #    lb_of_src, lb_of_tgt = data.y.squeeze(1)[src_edge_batch], data.y.squeeze(1)[tgt_edge_batch]
    #    batch_avg_homo = torch.sum(lb_of_src==lb_of_tgt).item() / len(lb_of_src)
    #    homo_of_hard.append(batch_avg_homo)


if __name__ == "__main__":
    main()
