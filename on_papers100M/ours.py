import os
import argparse

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from utils import *


def main():
    parser = argparse.ArgumentParser(description='OGBN-papers100M (MLP)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_sgc_iterations', type=int, default = 3)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_pick', type=int, default=16)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    sgc_dict = torch.load('/mnt/ogb_datasets/ogbn_papers100M/sgc_dict.pt')
    x = sgc_dict['sgc_embedding'][args.num_sgc_iterations]
    print(x.shape)
    split_idx = sgc_dict['split_idx']
    train_idx = split_idx['train'].to(device)
    test_idx = split_idx['test'].to(device)
    print(len(train_idx), len(split_idx['valid']), len(test_idx))

    train_zs = x[split_idx['train']].to(device)
    test_zs = x[split_idx['test']].to(device)
    #test_idx = split_idx['test'].to(device)

    def calc_dist(a, b):
        pairwise_dist = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1).sqrt()
        return pairwise_dist

    pagerank = load_rank_list('centrality')

    no_inc = 0
    relaxed = False
    rank = [pagerank[0]]
    tr_flag = torch.ones_like(train_idx)
    tr_flag[rank[0]] = 0
    tr_flag = tr_flag.bool()
    cur_dist = calc_dist(train_zs[rank[0]].unsqueeze(0), test_zs).squeeze(0)
    ts_flag = torch.ones_like(test_idx).bool()
    pbar = tqdm(total=len(train_idx))

    while len(rank) < len(train_idx):
        considered_dist = ts_flag * cur_dist
        batch_val, batch_idx = torch.topk(considered_dist, args.num_pick)
        valid_query_idx_mask = ts_flag[batch_idx]
        batch_val = batch_val[valid_query_idx_mask]
        batch_idx = batch_idx[valid_query_idx_mask]
        if len(batch_idx) == 0:
            # refresh
            print("Refresh test flag at 1.0!")
            ts_flag = torch.ones_like(test_idx).bool()
            init_tr_idx = -1
            for pr_idx in pagerank:
                if tr_flag[pr_idx]:
                    init_tr_idx = pr_idx
                    break
            assert init_tr_idx != -1, "No left for initiation!!!"
            print("Starting from {}".format(init_tr_idx))
            cur_dist = calc_dist(train_zs[init_tr_idx].unsqueeze(0), test_zs).squeeze(0)
            tr_flag[init_tr_idx] = False
            relaxed = False
            continue

        query_zs = test_zs[batch_idx]
        key_zs = train_zs[tr_flag]
        pairwise_dist = calc_dist(key_zs, query_zs)
        min_dist, min_dist_idx = pairwise_dist.min(0)

        if relaxed:
            pick_idx = tr_flag.nonzero().squeeze(1)[min_dist_idx]
            picked_dist = calc_dist(train_zs[pick_idx], test_zs)
            useful_idx = (picked_dist < cur_dist).sum(axis=1).nonzero().squeeze(1)
            pick_idx = pick_idx[useful_idx]
            pick_tr_idx = list(set(pick_idx.cpu().tolist()))
            rank.extend(pick_tr_idx)
            tr_flag[pick_idx] = False
            if len(pick_tr_idx) > 0:
                cur_dist = torch.minimum(cur_dist, picked_dist.min(0)[0])
        else:
            useful_flag = min_dist < batch_val
            pick_idx = min_dist_idx[useful_flag]
            pick_idx = tr_flag.nonzero().squeeze(1)[pick_idx]
            pick_tr_idx = list(set(train_idx[pick_idx].cpu().tolist()))
            rank.extend(pick_tr_idx)
            tr_flag[pick_idx] = False
            #ts_flag[batch_idx[torch.logical_not(useful_flag)]] = False
            if len(pick_tr_idx) > 0: 
                cur_dist = torch.minimum(cur_dist, calc_dist(train_zs[pick_idx], test_zs).min(0)[0])

        ts_flag[batch_idx] = False

        if len(pick_tr_idx) > 0:
            no_inc = 0
        else:
            no_inc += 1
            if no_inc >= args.patience:
                if len(rank) >= 0.9 * len(train_idx):
                    print("Greedily picked {} training nodes".format(len(rank)))
                    rank_as_set = set(rank)
                    other_nodes = [vid for vid in train_idx.cpu().tolist() if vid not in rank_as_set]
                    np.random.shuffle(other_nodes)
                    rank.extend(other_nodes)
                    break
                else:
                    print("Early no inc happens at {}!".format(len(rank)/float(len(train_idx))))
                    if not relaxed:
                        relaxed = True
                        no_inc = 0
                        print("relax selection criterion!")
                    else:
                        print("Refresh test flag at {}!".format( (torch.logical_not(ts_flag)).sum().cpu().item() / float(len(ts_flag)) ))
                        ts_flag = torch.ones_like(test_idx).bool()
                        init_tr_idx = -1
                        for pr_idx in pagerank:
                            if tr_flag[pr_idx]:
                                init_tr_idx = pr_idx
                                break
                        assert init_tr_idx != -1, "No left for initiation!!!"
                        print("Starting from {}".format(init_tr_idx))
                        cur_dist = calc_dist(train_zs[init_tr_idx].unsqueeze(0), test_zs).squeeze(0)
                        tr_flag[init_tr_idx] = False
                        relaxed = False
                        no_inc = 0

        pbar.update(len(pick_tr_idx))

    pbar.close()

    with open(os.path.join("ours", "train1.tsv"), 'w') as ops:
        for i in range(len(rank)):
            ops.write("{}\t{}\n".format(rank[i], i))


if __name__ == "__main__":
    main()
