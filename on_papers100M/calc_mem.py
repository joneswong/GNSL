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
        y_pred.append(out.argmax(dim=-1, keepdim=True).cpu())
        y_true.append(y)

    y_pred = torch.cat(y_pred, 0)
    y_true = torch.cat(y_true, 0)
    return (y_pred == y_true).squeeze(1)


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
    parser.add_argument('--end_sample_id', type=int, default=800)
    parser.add_argument('--fold', type=str, default='/mnt/ogb_datasets/ogbn_papers100M/ckpts')
    parser.add_argument('--infl', type=str, default='')
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
            train_cor = infer(model, device, train_loader)
            valid_cor= infer(model, device, valid_loader)
            test_cor = infer(model, device, test_loader)
            results[str(t)+"_train"] = train_cor
            results[str(t)+"_valid"] = valid_cor
            results[str(t)+"_test"] = test_cor
        torch.save(results, os.path.join(args.fold, "{}-{}.pth".format(args.start_sample_id, args.end_sample_id)))
    else:
        train_masks = []
        for t in tqdm(range(args.start_sample_id, args.end_sample_id)):
            content = torch.load(os.path.join(args.fold, "{}.pt".format(t)))
            mask = torch.zeros_like(split_idx['train'], dtype=torch.bool).scatter_(0, content['train_idx'], True)
            train_masks.append(mask)
        trainset_mask = np.vstack(train_masks)
        inv_mask = np.logical_not(trainset_mask)

        cors = dict()
        for fn in os.listdir(args.fold):
            if fn.endswith(".pth"):
                content = torch.load(os.path.join(args.fold, fn))
                cors.update(content)
        correctness = (len(cors) // 3) * [None]
        test_correctness = (len(cors) // 3) * [None]
        for k, v in cors.items():
            if k.endswith('train'):
                correctness[int(k[:k.find('_')])] = v.numpy()
            elif k.endswith('test'):
                test_correctness[int(k[:k.find('_')])] = v.numpy()

        correctness = np.vstack(correctness)
        test_correctness = np.vstack(test_correctness)

        if args.infl == 'max':
            test_correctness = torch.from_numpy(test_correctness).float().to(device)
            # (M, Tr)
            trainset_mask = torch.from_numpy(trainset_mask).float().to(device)
            inv_mask = torch.from_numpy(inv_mask).float().to(device)
            eps = torch.zeros_like(torch.sum(trainset_mask, axis=0, keepdims=True)) + 1e-8
            test_indices = torch.arange(len(split_idx['test'])).to(device)
            max_infl = torch.zeros_like(split_idx['train'], dtype=torch.float32).to(device) - 1.5

            def _masked_dot(x, mask, esp=1e-8):
                return torch.matmul(x, mask) / torch.maximum(torch.sum(mask, axis=0, keepdims=True), esp)

            for i in tqdm(range(0, len(test_indices), 16)):
                batch_test_indices = test_indices[i:min(i+16, len(test_indices))]

                # (B, M)
                batch_test_correctness = test_correctness.T[batch_test_indices]

                # (B, Tr)
                batch_infl_est = _masked_dot(batch_test_correctness, trainset_mask, eps) - _masked_dot(batch_test_correctness, inv_mask, eps)

                # (Tr, )
                batch_max_infl = torch.max(batch_infl_est, 0)[0]

                max_infl = torch.maximum(max_infl, batch_max_infl)

            indices = split_idx['train'].numpy().tolist()
            results = [(indices[i], float(max_infl[indices[i]].item())) for i in range(len(indices))]
            results = sorted(results, key=lambda x:x[1], reverse=True)
            with open(os.path.join("infl-{}".format(args.infl), "train.tsv"), 'w') as ops:
                for i in range(len(results)):
                    idx, infl = results[i]
                    ops.write("{}\t{}\n".format(idx, infl))
        elif args.infl == 'sum-abs':
            raise NotImplementedError("Don't use this")
        else:
            def _masked_avg(x, mask, axis=0, esp=1e-10):
                return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)

            def _masked_dot(x, mask, esp=1e-10):
                x = x.T.astype(np.float32)
                return (np.matmul(x, mask) / np.maximum(np.sum(mask, axis=0, keepdims=True), esp)).astype(np.float32)

            mem_est = _masked_avg(correctness, trainset_mask) - _masked_avg(correctness, inv_mask)
            #indices = data.train_mask.nonzero().squeeze(-1).numpy()
            indices = split_idx['train'].numpy().tolist()
            results = [(indices[i], float(mem_est[indices[i]])) for i in range(len(indices))]
            results = sorted(results, key=lambda x:x[1], reverse=True)
            with open(os.path.join("mem", "train.tsv"), 'w') as ops:
                for i in range(len(results)):
                    idx, mem = results[i]
                    ops.write("{}\t{}\n".format(idx, mem))


if __name__ == "__main__":
    main()
