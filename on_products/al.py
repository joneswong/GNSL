import torch

def select_by_mem(indices, alpha):
    num_rsv = alpha * len(indices)
    lookup = set(indices.numpy())
    new_indices = []
    with open("mem/train_mem.tsv", 'r') as ips:
        for line in ips:
            idx, _ = line.strip().split('\t')
            idx = int(idx)
            if idx in lookup:
                new_indices.append(int(idx))
            if len(new_indices) >= num_rsv:
                break

    if len(new_indices) < num_rsv:
        raise ValueError("Invalid length {} < {}".format(len(new_indices), num_rsv))

    return torch.Tensor(new_indices).long()
