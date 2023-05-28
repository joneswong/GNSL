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

    return F.softmax(out, dim=1)


def main():
    parser = argparse.ArgumentParser(description='Calculate Memorization')
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
            prob = infer(model, data, subgraph_loader, device)
            results[str(t)] = prob[torch.arange(prob.shape[0]), data.y.squeeze(1)]
        torch.save(results, os.path.join(args.fold, "{}-{}.pth".format(args.start_sample_id, args.end_sample_id)))
    else:
        train_masks = []
        for t in tqdm(range(args.start_sample_id, args.end_sample_id)):
            content = torch.load(os.path.join(args.fold, "{}.pt".format(t)), map_location=device)
            train_masks.append(content['train_mask'])

        probs = dict()
        for fn in os.listdir(args.fold):
            if fn.endswith(".pth"):
                content = torch.load(os.path.join(args.fold, fn), map_location=device)
                probs.update(content)

        labels = data.y.to(device).squeeze(1)
        train_cnts = torch.zeros_like(labels).long()
        train_mems = torch.zeros_like(labels, dtype=torch.float32)
        for i in tqdm(range(len(probs))):
            for j in range(len(probs)):
                if i == j:
                    continue
                gap = probs[str(i)] - probs[str(j)]
                flag = torch.logical_and(train_masks[i], torch.logical_not(train_masks[j]))
                train_cnts += flag
                train_mems += flag.float() * gap

        train_mems = (train_mems / (train_cnts + 1e-8)).cpu()
        train_mems = train_mems[data.train_mask].numpy()
        indices = data.train_mask.nonzero().squeeze(-1).numpy()
        results = [(int(indices[i]), float(train_mems[i])) for i in range(len(train_mems))]
        results = sorted(results, key=lambda x:x[1], reverse=True)
        with open(os.path.join(args.fold, "train_mem.tsv"), 'w') as ops:
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
