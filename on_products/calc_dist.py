import os
import argparse

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--mode', type=int, default=1)
    parser.add_argument('--algo', type=str, default='min')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_pick', type=int, default=8)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products',
                                         transform=T.ToSparseTensor())
    data = dataset[0]

    if args.mode == 0:
        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-1.0)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t
        data.adj_t = adj_t

        data = data.to(device)

        conv = GCNConv(data.x.shape[-1], data.x.shape[-1], normalize=False, bias=False)
        conv.to(device)
        #conv.reset_parameters()
        conv.lin.weight.data = torch.eye(data.x.shape[-1], device=device)

        @torch.no_grad()
        def prop(f, x, adj_t):
            for i in range(args.num_layers):
                x = f(x, adj_t)
            return x

        x = prop(conv, data.x, data.adj_t)

        torch.save(x, os.path.join("dist", "rw_feat.pt"))
    else:
        zs = torch.load(os.path.join("dist", "rw_feat.pt"))
        #print(zs.shape)
        #print(zs[123])

        if args.algo == 'min':
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train'].to(device)
            test_idx = split_idx['test'].to(device)
            train_zs = zs[train_idx]
            test_zs = zs[test_idx]

            min_dist = torch.Tensor(len(train_idx) * [99999]).to(device)
            for i in tqdm(range(0, len(test_idx), args.batch_size)):
                batch_test_idx = test_idx[i:min(i+args.batch_size, len(test_idx))]
                batch_test_zs = zs[batch_test_idx]
                batch_dist = ((train_zs.unsqueeze(1) - batch_test_zs.unsqueeze(0)) ** 2).sum(-1)
                batch_min_dist, _ = batch_dist.min(-1)
                min_dist = torch.minimum(min_dist, batch_min_dist)

            train_idx = train_idx.cpu().tolist()
            min_dist = min_dist.cpu().tolist()
            results = [(i, vi) for i, vi in zip(train_idx, min_dist)]
            results = sorted(results, key=lambda x:x[1])
            with open(os.path.join("dist-min", "train.tsv"), 'w') as ops:
                for i in range(len(results)):
                    idx, dist = results[i]
                    ops.write("{}\t{}\n".format(idx, dist))
        elif args.algo == 'sum':
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train'].to(device)
            test_idx = split_idx['test'].to(device)
            train_zs = zs[train_idx]
            test_zs = zs[test_idx]

            sum_dist = torch.zeros_like(train_idx)
            for i in tqdm(range(0, len(test_idx), args.batch_size)):
                batch_test_idx = test_idx[i:min(i+args.batch_size, len(test_idx))]
                batch_test_zs = zs[batch_test_idx]
                batch_dist = ((train_zs.unsqueeze(1) - batch_test_zs.unsqueeze(0)) ** 2).sum(-1).sqrt()
                batch_sum_dist = batch_dist.sum(-1)
                sum_dist = sum_dist + batch_sum_dist

            train_idx = train_idx.cpu().tolist()
            sum_dist = sum_dist.cpu().tolist()
            results = [(i, vi) for i, vi in zip(train_idx, sum_dist)]
            results = sorted(results, key=lambda x:x[1])
            with open(os.path.join("dist-sum", "train.tsv"), 'w') as ops:
                for i in range(len(results)):
                    idx, dist = results[i]
                    ops.write("{}\t{}\n".format(idx, dist))
        elif args.algo == 'min-cnt':
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train'].to(device)
            test_idx = split_idx['test'].to(device)
            train_zs = zs[train_idx]
            test_zs = zs[test_idx]

            cnt_min_dist = torch.zeros_like(train_idx)
            for i in tqdm(range(0, len(test_idx), args.batch_size)):
                batch_test_idx = test_idx[i:min(i+args.batch_size, len(test_idx))]
                batch_test_zs = zs[batch_test_idx]
                batch_dist = ((train_zs.unsqueeze(1) - batch_test_zs.unsqueeze(0)) ** 2).sum(-1).sqrt()
                _, batch_min_dist_idx = batch_dist.min(-1)
                cnt_min_dist[batch_min_dist_idx] += 1

            train_idx = train_idx.cpu().tolist()
            cnt_min_dist = cnt_min_dist.cpu().tolist()
            results = [(i, vi) for i, vi in zip(train_idx, cnt_min_dist)]
            results = sorted(results, key=lambda x:x[1], reverse=True)
            with open(os.path.join("dist-min-cnt", "train.tsv"), 'w') as ops:
                for i in range(len(results)):
                    idx, dist = results[i]
                    ops.write("{}\t{}\n".format(idx, dist))
        elif args.algo == 'greedy':
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train'].to(device)
            test_idx = split_idx['test'].to(device)
            train_zs = zs[train_idx]
            test_zs = zs[test_idx]

            def calc_dist(a, b):
                pairwise_dist = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(-1).sqrt()
                return pairwise_dist

            no_inc = 0
            rank = [100864]
            tr_flag = torch.ones_like(train_idx)
            tr_flag[rank[0]] = 0
            tr_flag = tr_flag.bool()
            cur_dist = calc_dist(zs[100864].unsqueeze(0), test_zs).squeeze(0)
            ts_flag = torch.ones_like(test_idx).bool()
            pbar = tqdm(total=len(train_idx))

            while len(rank) < len(train_idx) and (torch.logical_not(ts_flag)).sum().cpu().item() < len(test_idx):
                considered_dist = ts_flag * cur_dist
                batch_val, batch_idx = torch.topk(considered_dist, args.num_pick)
                query_zs = test_zs[batch_idx]
                key_zs = train_zs[tr_flag]
                pairwise_dist = calc_dist(key_zs, query_zs)
                min_dist, min_dist_idx = pairwise_dist.min(0)
                useful_flag = min_dist < batch_val
                pick_idx = min_dist_idx[useful_flag]
                pick_idx = tr_flag.nonzero().squeeze(1)[pick_idx]
                pick_tr_idx = list(set(train_idx[pick_idx].cpu().tolist()))
                rank.extend(pick_tr_idx)
                tr_flag[pick_idx] = False
                ts_flag[batch_idx[torch.logical_not(useful_flag)]] = False
                if len(pick_idx) > 0:
                    cur_dist = torch.minimum(cur_dist, calc_dist(train_zs[pick_idx], test_zs).min(0)[0])
                    no_inc = 0
                else:
                    no_inc += 1
                    if no_inc >= args.patience:
                        if len(rank) >= 0.8 * len(train_idx):
                            print("Greedily picked {} training nodes".format(len(rank)))
                            rank_as_set = set(rank)
                            other_nodes = [vid for vid in train_idx.cpu().tolist() if vid not in rank_as_set]
                            rank.extend(other_nodes)
                            break
                        else:
                            print("Early no inc!")
                pbar.update(len(pick_tr_idx))

            pbar.close()

            #train_idx = train_idx.cpu().tolist()
            #cnt_min_dist = cnt_min_dist.cpu().tolist()
            #results = [(i, vi) for i, vi in zip(train_idx, cnt_min_dist)]
            #results = sorted(results, key=lambda x:x[1], reverse=True)
            with open(os.path.join("dist-greedy", "train.tsv"), 'w') as ops:
                for i in range(len(rank)):
                    ops.write("{}\t{}\n".format(rank[i], i))
        else:
            raise ValueError(args.algo)


if __name__ == "__main__":
    main()
