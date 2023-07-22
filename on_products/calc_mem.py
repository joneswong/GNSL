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

from ogb.nodeproppred import PygNodePropPredDataset


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
    y_true = data.y
    y_pred = out.argmax(dim=-1, keepdim=True)

    return (y_pred == y_true).squeeze(1)


def main():
    parser = argparse.ArgumentParser(description='Calculate Memorization and Influence')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=20000)
    # exp relatead
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--start_sample_id', type=int, default=0)
    parser.add_argument('--end_sample_id', type=int, default=1000)
    parser.add_argument('--fold', type=str, default='mem')
    parser.add_argument('--infl', type=str, default='')
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

    if args.mode == 0:
        model = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                     args.num_layers, args.dropout).to(device)

        subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                          batch_size=4096, shuffle=False,
                                          num_workers=12)

        #train_masks, valid_masks, probs = [], [], []
        results = dict()
        for t in tqdm(range(args.start_sample_id, args.end_sample_id)):
            content = torch.load(os.path.join(args.fold, "{}.pt".format(t)), map_location=device)
            model.load_state_dict(content['model'])
            #train_masks.append(content['train_mask'])
            #valid_masks.append(content['valid_mask'])
            #probs.append(infer(model, data, subgraph_loader, device))
            cor = infer(model, data, subgraph_loader, device)
            results[str(t)] = cor
        torch.save(results, os.path.join(args.fold, "{}-{}.pth".format(args.start_sample_id, args.end_sample_id)))
    else:
        train_masks = []
        for t in tqdm(range(args.start_sample_id, args.end_sample_id)):
            content = torch.load(os.path.join(args.fold, "{}.pt".format(t)))
            train_masks.append(content['train_mask'].numpy())
        trainset_mask = np.vstack(train_masks)
        inv_mask = np.logical_not(trainset_mask)

        cors = dict()
        for fn in os.listdir(args.fold):
            if fn.endswith(".pth"):
                content = torch.load(os.path.join(args.fold, fn))
                cors.update(content)
        correctness = len(cors) * [None]
        for k, v in cors.items():
            correctness[int(k)] = v.numpy()
        correctness = np.vstack(correctness)


        if args.infl == 'max':
            correctness = torch.from_numpy(correctness).float().to(device)
            trainset_mask = torch.from_numpy(trainset_mask).float().to(device)
            inv_mask = torch.from_numpy(inv_mask).float().to(device)
            eps = torch.zeros_like(torch.sum(trainset_mask, axis=0, keepdims=True)) + 1e-8
            test_indices = split_idx['test'].to(device)
            max_infl = torch.zeros_like(data.train_mask, dtype=torch.float32).to(device) - 1.5
            def _masked_dot(x, mask, esp=1e-8):
                return torch.matmul(x, mask) / torch.maximum(torch.sum(mask, axis=0, keepdims=True), esp)

            for i in tqdm(range(0, len(test_indices), 16)):
                batch_test_indices = test_indices[i:min(i+16, len(test_indices))]
                batch_test_correctness = correctness.T[batch_test_indices]
                batch_infl_est = _masked_dot(batch_test_correctness, trainset_mask, eps) - _masked_dot(batch_test_correctness, inv_mask, eps)
                batch_max_infl = torch.max(batch_infl_est, 0)[0]
                max_infl = torch.maximum(max_infl, batch_max_infl)

            indices = data.train_mask.nonzero().squeeze(-1).numpy()
            results = [(indices[i], float(max_infl[indices[i]].item())) for i in range(len(indices))]
            results = sorted(results, key=lambda x:x[1], reverse=True)
            with open(os.path.join("infl-{}".format(args.infl), "train.tsv"), 'w') as ops:
                for i in range(len(results)):
                    idx, infl = results[i]
                    ops.write("{}\t{}\n".format(idx, infl))
        elif args.infl == 'sum-abs':
            correctness = torch.from_numpy(correctness).float().to(device)
            trainset_mask = torch.from_numpy(trainset_mask).float().to(device)
            inv_mask = torch.from_numpy(inv_mask).float().to(device)
            eps = torch.zeros_like(torch.sum(trainset_mask, axis=0, keepdims=True)) + 1e-8
            test_indices = split_idx['test'].to(device)
            sum_abs_infl = torch.zeros_like(data.train_mask, dtype=torch.float32).to(device)
            def _masked_dot(x, mask, esp=1e-8):
                return torch.matmul(x, mask) / torch.maximum(torch.sum(mask, axis=0, keepdims=True), esp)

            for i in tqdm(range(0, len(test_indices), 16)):
                batch_test_indices = test_indices[i:min(i+16, len(test_indices))]
                batch_test_correctness = correctness.T[batch_test_indices]
                batch_infl_est = _masked_dot(batch_test_correctness, trainset_mask, eps) - _masked_dot(batch_test_correctness, inv_mask, eps)
                batch_sum_abs_infl = torch.sum(torch.abs(batch_infl_est), 0)
                sum_abs_infl += batch_sum_abs_infl

            indices = data.train_mask.nonzero().squeeze(-1).numpy()
            results = [(indices[i], float(sum_abs_infl[indices[i]].item())) for i in range(len(indices))]
            results = sorted(results, key=lambda x:x[1], reverse=True)
            with open(os.path.join("infl-{}".format(args.infl), "train.tsv"), 'w') as ops:
                for i in range(len(results)):
                    idx, infl = results[i]
                    ops.write("{}\t{}\n".format(idx, infl))
        else:
            def _masked_avg(x, mask, axis=0, esp=1e-10):
                return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

            def _masked_dot(x, mask, esp=1e-10):
                x = x.T.astype(np.float32)
                return (np.matmul(x, mask) / np.maximum(np.sum(mask, axis=0, keepdims=True), esp)).astype(np.float32)

            mem_est = _masked_avg(correctness, trainset_mask) - _masked_avg(correctness, inv_mask)
            indices = data.train_mask.nonzero().squeeze(-1).numpy()
            results = [(indices[i], float(mem_est[indices[i]])) for i in range(len(indices))]
            results = sorted(results, key=lambda x:x[1], reverse=True)
            with open(os.path.join(args.fold, "train.tsv"), 'w') as ops:
                for i in range(len(results)):
                    idx, mem = results[i]
                    ops.write("{}\t{}\n".format(idx, mem))
        """
        for fn in os.listdir(args.fold):
            if fn.endswith(".pt"):
                content = torch.load(os.path.join(args.fold, fn))
                new_content = dict()
                for k, v in content.items():
                    new_content[k] = v[torch.arange(v.shape[0]), data.y.squeeze(1)]
                torch.save(new_content, os.path.join(args.fold, fn.replace(".pt", ".pth")))
        """


if __name__ == "__main__":
    main()
