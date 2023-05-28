from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import torch
from torch_geometric.nn import SAGEConv


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


dataset = PygNodePropPredDataset(name='ogbn-products')
split_idx = dataset.get_idx_split()
print(split_idx)
print(split_idx['valid'], len(split_idx['valid']))
print(split_idx['train'], len(split_idx['train']))
data = dataset[0]
print(data.y.shape)


"""
s0 = torch.load("mem/0.pt")
s1 = torch.load("mem/1.pt")
train_mask0 = s0['train_mask']
train_mask1 = s1['train_mask']
print(train_mask0, train_mask0.device)
print(train_mask1, train_mask1.device)

device = f'cuda:0'
device = torch.device(device)

model = SAGE(data.x.size(-1), 256, dataset.num_classes,
             3, 0.5).to(device)
for p in model.parameters():
    print(p)
    break
model.load_state_dict(s0['model'])
print(model)
for p in model.parameters():
    print(p)
    break
"""
