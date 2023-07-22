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

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

from utils import *


def get_samples(methods, ratio):
    ranks = []
    for fold in methods.split(','):
        ranks.append(aug_rank_by_pct(load_rank_list(fold)))
    ensembled_rank = dict(ranks[0])
    for i in range(1, len(ranks)):
        for tp in ranks[i]:
            ensembled_rank[tp[0]] += tp[1]
    ensembled_rank = sorted([(k, v) for k, v in ensembled_rank.items()], key=lambda x:x[1], reverse=True)

    return [tp[0] for tp in ensembled_rank[:int(ratio*len(ensembled_rank))]], [tp[0] for tp in ensembled_rank[-int(ratio*len(ensembled_rank)):]]


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
    # exp relatead
    parser.add_argument('--al', type=str, default='mem,infl-max,infl-sum-abs')
    parser.add_argument('--ratio', type=float, default=0.25)
    args = parser.parse_args()
    print(args)

    dataset = PygNodePropPredDataset(name='ogbn-products')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    hard_sample, easy_sample = get_samples(args.al, args.ratio)

    # degree
    degrees = pyg.utils.degree(data.edge_index[0], data.num_nodes).numpy()
    degrees_of_hard = degrees[hard_sample]
    degrees_of_easy = degrees[easy_sample]
    mean_of_hard, std_of_hard = np.mean(degrees_of_hard), np.std(degrees_of_hard)
    mean_of_easy, std_of_easy = np.mean(degrees_of_easy), np.std(degrees_of_easy)
    print("{} ({}) v.s. {} ({})".format(mean_of_hard, std_of_hard, mean_of_easy, std_of_easy))

    # page rank score
    pr = torch.load(os.path.join('age', "pr.pt"))
    pr = np.asarray([len(pr) * pr[i] for i in range(len(pr))])
    pr_of_hard = pr[hard_sample]
    pr_of_easy = pr[easy_sample]
    mean_of_hard, std_of_hard = np.mean(pr_of_hard), np.std(pr_of_hard)
    mean_of_easy, std_of_easy = np.mean(pr_of_easy), np.std(pr_of_easy)
    print("{} ({}) v.s. {} ({})".format(mean_of_hard, std_of_hard, mean_of_easy, std_of_easy))

    # chaotic neighborhood
    # avg homophilic level
    lb_of_src, lb_of_tgt = data.y.squeeze(1)[data.edge_index[0]], data.y.squeeze(1)[data.edge_index[1]]
    print("Homophilic level of this graph is {}".format( torch.sum(lb_of_src==lb_of_tgt).item() / len(lb_of_src) ))
    # homophilic levels of specific node(s)
    homo_of_hard = []
    for i in tqdm(range(0, len(hard_sample), 32)):
        node_batch = hard_sample[i:min(i+32, len(hard_sample))]
        if len(node_batch) < 32:
            break
        idx = (data.edge_index[0].unsqueeze(1) == torch.Tensor(node_batch).long().unsqueeze(0)).nonzero()
        idx_ = idx[:,0]
        src_edge_batch, tgt_edge_batch = data.edge_index[0][idx_], data.edge_index[1][idx_]
        lb_of_src, lb_of_tgt = data.y.squeeze(1)[src_edge_batch], data.y.squeeze(1)[tgt_edge_batch]
        batch_avg_homo = torch.sum(lb_of_src==lb_of_tgt).item() / len(lb_of_src)
        homo_of_hard.append(batch_avg_homo)
    homo_of_easy = []
    for i in tqdm(range(0, len(easy_sample), 32)):
        node_batch = easy_sample[i:min(i+32, len(easy_sample))]
        if len(node_batch) < 32:
            break
        idx = (data.edge_index[0].unsqueeze(1) == torch.Tensor(node_batch).long().unsqueeze(0)).nonzero()
        idx_ = idx[:,0]
        src_edge_batch, tgt_edge_batch = data.edge_index[0][idx_], data.edge_index[1][idx_]
        lb_of_src, lb_of_tgt = data.y.squeeze(1)[src_edge_batch], data.y.squeeze(1)[tgt_edge_batch]
        batch_avg_homo = torch.sum(lb_of_src==lb_of_tgt).item() / len(lb_of_src)
        homo_of_easy.append(batch_avg_homo)
    mean_of_hard, std_of_hard = np.mean(homo_of_hard), np.std(homo_of_hard)
    mean_of_easy, std_of_easy = np.mean(homo_of_easy), np.std(homo_of_easy)
    print("{} ({}) v.s. {} ({})".format(mean_of_hard, std_of_hard, mean_of_easy, std_of_easy))


if __name__ == "__main__":
    main()
