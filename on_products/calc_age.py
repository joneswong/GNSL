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

import torch_geometric as pyg

from ogb.nodeproppred import PygNodePropPredDataset

from scipy import stats


def val2pct(vals):
    arr = [(i, vals[i]) for i in range(len(vals))]
    arr = sorted(arr, key=lambda x:x[1], reverse=True)
    pct = len(arr) * [None]
    for i in range(len(arr)):
        pct[arr[i][0]] = (len(arr) - i) / float(len(arr))
    return np.asarray(pct)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return torch.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


@torch.no_grad()
def infer(model, data, subgraph_loader, device):
    model.eval()

    out = model.inference(data.x, subgraph_loader, device)

    return F.softmax(out, dim=1)


def main():
    parser = argparse.ArgumentParser(description='Calculate AGE')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=20000)
    # exp relatead
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--metric', type=str, default='uncertainty')
    parser.add_argument('--alpha', type=float, default=0.3333)
    parser.add_argument('--beta', type=float, default=0.3333)
    parser.add_argument('--fold', type=str, default='age')
    parser.add_argument('--output_fold', type=str, default='')
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

    model = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                 args.num_layers, args.dropout).to(device)

    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=4096, shuffle=False,
                                      num_workers=12)

    if args.mode == 0:
        if args.metric == 'uncertainty':
            results = dict()
            for i in range(3):
                content = torch.load(os.path.join(args.fold, "{}.pt".format(i)), map_location=device)
                model.load_state_dict(content['model'])
                prob = infer(model, data, subgraph_loader, device)
                m = torch.distributions.categorical.Categorical(probs=prob)
                uncertainty = m.entropy()
                results[str(i)] = uncertainty
            torch.save(results, os.path.join(args.fold, 'uncertainty.pt'))
        elif args.metric == 'centrality':
            pass
        elif args.metric == 'density':
            content = torch.load(os.path.join(args.fold, "4.pt"), map_location=device)
            model.load_state_dict(content['model'])
            prob = infer(model, data, subgraph_loader, device).numpy()
            # k-means
            from sklearn.cluster import KMeans
            from sklearn.metrics.pairwise import euclidean_distances
            NCL = dataset.num_classes // 2
            kmeans = KMeans(n_clusters=NCL, random_state=42).fit(prob)
            ed = euclidean_distances(prob, kmeans.cluster_centers_)
            ed_score = np.min(ed, axis=1)#the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is
            #edprec = np.asarray([percd(ed_score, i) for i in range(len(ed_score))])
            torch.save(torch.from_numpy(ed_score), os.path.join(args.fold, 'density.pt'))
        else:
            raise ValueError("No this metric now")
    else:
        pr = torch.load(os.path.join(args.fold, "pr.pt"))
        pr = np.asarray([len(pr) * pr[i] for i in range(len(pr))])
        #D = pyg.utils.degree(data.edge_index[0], num_nodes=data.x.size(0)).cpu().numpy()
        #statistic, pval = stats.spearmanr(pr, D)
        #print(statistic)
        uncertainty = torch.load(os.path.join(args.fold, "uncertainty.pt"))
        u = .0
        for k, v in uncertainty.items():
            u += v
        u = u / len(uncertainty)
        d = torch.load(os.path.join(args.fold, "density.pt"))
        print(d.shape)

        if args.metric == 'centrality':
            metric = pr
        elif args.metric == 'uncertainty':
            metric = u
        elif args.metric == 'density':
            metric = d
        elif args.metric == 'age':
            pr = val2pct(pr)
            u = val2pct(u)
            d = val2pct(d)
            metric = args.alpha * u + args.beta * pr + (1.0 - args.beta - args.alpha) * d
        else:
            raise NotImplementedError("No this metric now")

        results = [(i, float(metric[i].item())) for i in range(len(data['train_mask'])) if data['train_mask'][i] == True]
        results = sorted(results, key=lambda x:x[1], reverse=True)
        if args.output_fold:
            output_fold = args.output_fold
        else:
            output_fold = args.metric
        with open(os.path.join(output_fold, "train.tsv"), 'w') as ops:
            for i in range(len(results)):
                idx, norm = results[i]
                ops.write("{}\t{}\n".format(idx, norm))


if __name__ == "__main__":
    main()
