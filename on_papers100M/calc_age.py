import os
import argparse
import time

from tqdm.auto import tqdm
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph, to_undirected

from torch_sparse import SparseTensor
import torch_geometric as pyg

from ogb.nodeproppred import PygNodePropPredDataset

import scipy as sp
from scipy import stats


def my_pagerank_scipy(
    N,
    A,
    alpha=0.85,
    max_iter=100,
    tol=1.0e-10,
    weight="weight",
):
    if N == 0:
        return {}

    nodelist = list(range(N))
    print(A.shape)
    S = A.sum(axis=1)
    print(S.shape)
    S[S != 0] = 1.0 / S[S != 0]
    # TODO: csr_array
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    x = np.repeat(1.0 / N, N)

    # Personalization vector
    p = np.repeat(1.0 / N, N)
    
    # Dangling nodes
    dangling_weights = p
    is_dangling = np.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in tqdm(range(max_iter)):
        xlast = x
        x = alpha * (x @ A + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        print(err)
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)


def val2pct(vals):
    arr = [(i, vals[i]) for i in range(len(vals))]
    arr = sorted(arr, key=lambda x:x[1], reverse=True)
    pct = len(arr) * [None]
    for i in range(len(arr)):
        pct[arr[i][0]] = (len(arr) - i) / float(len(arr))
    return np.asarray(pct)


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert self.x.size(0) == self.y.size(0)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


@torch.no_grad()
def infer(model, device, loader):
    model.eval()

    y_pred = []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        y_pred.append(torch.exp(out))

    y_pred = torch.cat(y_pred, 0)
    return y_pred


def export_rank(metric, output_fold):
    results = [(i, metric[i]) for i in range(len(metric))]
    results = sorted(results, key=lambda x:x[1], reverse=True)
    with open(os.path.join(output_fold, "train.tsv"), 'w') as ops:
        for i in range(len(results)):
            idx, val = results[i]
            ops.write("{}\t{}\n".format(idx, val))


def main():
    parser = argparse.ArgumentParser(description='Calculate AGE')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--use_sgc_embedding', action='store_true')
    parser.add_argument('--num_sgc_iterations', type=int, default = 3)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
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

    if args.mode == 0:
        if args.metric == 'uncertainty':
            if not args.use_sgc_embedding:
                try:
                    data_dict = torch.load('data_dict.pt')
                except:
                    raise RuntimeError('data_dict.pt not found. Need to run python node2vec.py first')
                x = data_dict['node_feat']
                split_idx = data_dict['split_idx']
                y = data_dict['label'].to(torch.long)

                if args.use_node_embedding:
                    x = torch.cat([x, data_dict['node2vec_embedding']], dim=-1)

                print(x.shape)
            else:
                if args.use_node_embedding:
                    raise ValueError('No option to use node embedding and sgc embedding at the same time.')
                else:
                    try:
                        sgc_dict = torch.load('/mnt/ogb_datasets/ogbn_papers100M/sgc_dict.pt')
                    except:
                        raise RuntimeError('sgc_dict.pt not found. Need to run python sgc.py first')

                    x = sgc_dict['sgc_embedding'][args.num_sgc_iterations]
                    split_idx = sgc_dict['split_idx']
                    y = sgc_dict['label'].to(torch.long)

            train_dataset = SimpleDataset(x[split_idx['train']], y[split_idx['train']])
            valid_dataset = SimpleDataset(x[split_idx['valid']], y[split_idx['valid']])
            test_dataset = SimpleDataset(x[split_idx['test']], y[split_idx['test']])

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


            model = MLP(x.size(-1), args.hidden_channels, 172, args.num_layers,
                        args.dropout).to(device)
            results = dict()
            for i in range(8):
                content = torch.load(os.path.join('/mnt/ogb_datasets/ogbn_papers100M/ckpts', "{}.pt".format(i)), map_location=device)
                model.load_state_dict(content['model'])
                train_prob = infer(model, device, train_loader)
                m = torch.distributions.categorical.Categorical(probs=train_prob)
                uncertainty = m.entropy()
                results[str(i)] = uncertainty
            torch.save(results, os.path.join('age', 'uncertainty.pt'))

        elif args.metric == 'centrality':

            N = 111059956
            
            edge_index = torch.load('/mnt/ogb_datasets/ogbn_papers100M/edge_index.pt')

            row, col = edge_index
            print(row.shape, col.shape)

            print('Computing adj...')

            #adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
            #adj = adj.set_diag()
            #deg = adj.sum(dim=1).to(torch.float)
            #deg_inv_sqrt = deg.pow(-0.5)
            #deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            #adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
            #adj = adj.to_scipy(layout='csr')

            adj = sp.sparse.coo_array((np.ones(len(row)), (row.numpy(), col.numpy())), shape=(N, N), dtype=float)
            adj = adj.asformat('csr')
            print(adj.shape, adj.dtype)

            # Pagerank score calculation
            start = time.time()
            pr = my_pagerank_scipy(N, adj)
            end = time.time()
            print(time.localtime(start), time.localtime(end))
            torch.save(pr, os.path.join('age', 'pr.pt'))

        elif args.metric == 'density':
            if not args.use_sgc_embedding:
                try:
                    data_dict = torch.load('data_dict.pt')
                except:
                    raise RuntimeError('data_dict.pt not found. Need to run python node2vec.py first')
                x = data_dict['node_feat']
                split_idx = data_dict['split_idx']
                y = data_dict['label'].to(torch.long)

                if args.use_node_embedding:
                    x = torch.cat([x, data_dict['node2vec_embedding']], dim=-1)

                print(x.shape)
            else:
                if args.use_node_embedding:
                    raise ValueError('No option to use node embedding and sgc embedding at the same time.')
                else:
                    try:
                        sgc_dict = torch.load('/mnt/ogb_datasets/ogbn_papers100M/sgc_dict.pt')
                    except:
                        raise RuntimeError('sgc_dict.pt not found. Need to run python sgc.py first')

                    x = sgc_dict['sgc_embedding'][args.num_sgc_iterations]
                    split_idx = sgc_dict['split_idx']
                    y = sgc_dict['label'].to(torch.long)

            train_dataset = SimpleDataset(x[split_idx['train']], y[split_idx['train']])
            valid_dataset = SimpleDataset(x[split_idx['valid']], y[split_idx['valid']])
            test_dataset = SimpleDataset(x[split_idx['test']], y[split_idx['test']])

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


            model = MLP(x.size(-1), args.hidden_channels, 172, args.num_layers,
                        args.dropout).to(device)

            content = torch.load(os.path.join('/mnt/ogb_datasets/ogbn_papers100M/ckpts', "0.pt"), map_location=device)
            model.load_state_dict(content['model'])
            train_prob = infer(model, device, train_loader)
            val_prob = infer(model, device, valid_loader)
            test_prob = infer(model, device, test_loader)
            prob = torch.cat([train_prob, val_prob, test_prob], axis=0).cpu().numpy()

            # k-means
            from sklearn.cluster import KMeans
            from sklearn.metrics.pairwise import euclidean_distances
            NCL = 172 // 2
            kmeans = KMeans(n_clusters=NCL, random_state=42).fit(prob)
            ed = euclidean_distances(prob, kmeans.cluster_centers_)
            ed_score = np.min(ed, axis=1)#the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is
            #edprec = np.asarray([percd(ed_score, i) for i in range(len(ed_score))])
            torch.save(torch.from_numpy(ed_score), os.path.join('age', 'density.pt'))
        else:
            raise ValueError("No this metric now")

    else:
        dataset = PygNodePropPredDataset('ogbn-papers100M', root="/mnt/ogb_datasets")
        split_idx = dataset.get_idx_split()
        pr = torch.load(os.path.join(args.fold, "pr.pt"))
        pr = np.asarray([len(pr) * pr[i] for i in range(len(pr))])
        pr = pr[split_idx['train'].numpy()].tolist()
        #D = pyg.utils.degree(data.edge_index[0], num_nodes=data.x.size(0)).cpu().numpy()
        #statistic, pval = stats.spearmanr(pr, D)
        #print(statistic)
        uncertainty = torch.load(os.path.join(args.fold, "uncertainty.pt"))
        u = .0
        for k, v in uncertainty.items():
            u += v
        u = u / len(uncertainty)
        u = u.cpu().numpy().tolist()
        d = torch.load(os.path.join(args.fold, "density.pt"))
        d = d[torch.arange(len(split_idx['train']))].numpy().tolist()

        export_rank(pr, "centrality")
        export_rank(u, "uncertainty")
        export_rank(d, "density")
        pr = val2pct(pr)
        u = val2pct(u)
        d = val2pct(d)
        metric = args.alpha * u + args.beta * pr + (1.0 - args.beta - args.alpha) * d
        export_rank(metric.tolist(), "age")


if __name__ == "__main__":
    main()
