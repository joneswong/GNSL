import os
import argparse
import copy
from tqdm.auto import tqdm

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ogb.nodeproppred import Evaluator

from logger import Logger


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

    y_pred, y_true = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        y_pred.append(torch.exp(out).cpu())
        y_true.append(y)

    y_pred = torch.cat(y_pred, 0)
    y_true = torch.cat(y_true, 0)
    return y_pred[torch.arange(y_pred.shape[0]), y_true.squeeze(1)]


def main():
    parser = argparse.ArgumentParser(description='OGBN-papers100M (MLP)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--use_sgc_embedding', action='store_true')
    parser.add_argument('--num_sgc_iterations', type=int, default = 3)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    # exp relatead
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--start_sample_id', type=int, default=0)
    parser.add_argument('--end_sample_id', type=int, default=1000)
    parser.add_argument('--fold', type=str, default='mem')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)



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
                sgc_dict = torch.load('sgc_dict.pt')
            except:
                raise RuntimeError('sgc_dict.pt not found. Need to run python sgc.py first')

            x = sgc_dict['sgc_embedding'][args.num_sgc_iterations]
            split_idx = sgc_dict['split_idx']
            y = sgc_dict['label'].to(torch.long)


    train_dataset = SimpleDataset(x[split_idx['train']], y[split_idx['train']])
    valid_dataset = SimpleDataset(x[split_idx['valid']], y[split_idx['valid']])
    test_dataset = SimpleDataset(x[split_idx['test']], y[split_idx['test']])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = MLP(x.size(-1), args.hidden_channels, 172, args.num_layers,
                args.dropout).to(device)

    if args.mode == 0:
        results = dict()
        for t in tqdm(range(args.start_sample_id, args.end_sample_id)):
            content = torch.load(os.path.join(args.fold, "{}.pt".format(t)), map_location=device)
            model.load_state_dict(content['model'])
            train_lbprob = infer(model, device, train_loader)
            valid_lbprob= infer(model, device, valid_loader)
            test_lbprob = infer(model, device, test_loader)
            results[str(t)+"_train"] = train_lbprob
            results[str(t)+"_valid"] = valid_lbprob
            results[str(t)+"_test"] = test_lbprob
        torch.save(results, os.path.join(args.fold, "{}-{}.pth".format(args.start_sample_id, args.end_sample_id)))
    else:
        split_idx['train'] = split_idx['train'].to(device)
        train_masks = []
        for t in tqdm(range(args.start_sample_id, args.end_sample_id)):
            content = torch.load(os.path.join(args.fold, "{}.pt".format(t)), map_location=device)
            mask = torch.zeros_like(split_idx['train'], dtype=torch.bool).scatter_(0, content['train_idx'], True)
            train_masks.append(mask)

        lbprobs = dict()
        for fn in os.listdir(args.fold):
            if fn.endswith(".pth"):
                content = torch.load(os.path.join(args.fold, fn), map_location=device)
                lbprobs.update(content)

        train_cnts = torch.zeros_like(split_idx['train']).long()
        train_mems = torch.zeros_like(split_idx['train'], dtype=torch.float32)
        for i in tqdm(range(args.end_sample_id - args.start_sample_id)):
            for j in range(args.end_sample_id - args.start_sample_id):
                if i == j:
                    continue
                gap = lbprobs[str(i)+"_train"] - lbprobs[str(j)+"_train"]
                flag = torch.logical_and(train_masks[i], torch.logical_not(train_masks[j]))
                train_cnts += flag
                train_mems += flag.float() * gap

        train_mems = (train_mems / (train_cnts + 1e-8)).cpu()
        indices = split_idx['train'].cpu().numpy()
        results = [(int(indices[i]), float(train_mems[i])) for i in range(len(train_mems))]
        results = sorted(results, key=lambda x:x[1], reverse=True)
        with open(os.path.join(args.fold, "train.tsv"), 'w') as ops:
            for i in range(len(results)):
                idx, mem = results[i]
                ops.write("{}\t{}\n".format(idx, mem))



if __name__ == "__main__":
    main()
