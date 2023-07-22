import os
import math
import random

import numpy as np
import torch


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_rank_list(al):
    ranks = []
    with open(os.path.join(al, "train.tsv"), 'r') as ips:
        for line in ips:
            idx, _ = line.strip().split('\t')
            idx = int(idx)
            ranks.append(idx)
    return ranks


def aug_rank_by_pct(rank):
    return [(idx, float(len(rank)-i)/len(rank)) for i, idx in enumerate(rank)]


#def index_to_mask(index, size):
#    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
#    mask[index] = 1
#    return mask


def random_splits(data, num_classes, idx, rsv):
    # * round(reserve_rate*len(data)/num_classes) * num_classes labels for training

    assert rsv >= .0 and rsv <= 1.0, "Invalid value {} for reserve_rate".format(rsv)

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[idx] = True

    indices = []
    new_lens = []
    for i in range(num_classes):
        index = torch.logical_and((data.y.squeeze(1) == i), mask).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

        expected_l = rsv * len(index)
        l = math.floor(expected_l)
        l = l + (1 if (expected_l - l) >= random.uniform(0, 1) else 0)
        new_lens.append(l)

    index = torch.cat([arr[:new_lens[i]] for i, arr in enumerate(indices)], dim=0)

    #if Flag is 0:
    #    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    #    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    #    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    #    data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
    #    data.test_mask = index_to_mask(
    #    rest_index[val_lb:], size=data.num_nodes)
    #else:
    #    val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
    #                           for i in indices], dim=0)
    #    rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
    #    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    #    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    #    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    #    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    #return data

    return index


def select_by_al(data, num_classes, idx, rsv, ranks):
    # * round(reserve_rate*len(data)/num_classes) * num_classes labels for training

    assert rsv >= .0 and rsv <= 1.0, "Invalid value {} for reserve_rate".format(rsv)

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[idx] = True

    indices = []
    new_lens = []
    for i in range(num_classes):
        index = torch.logical_and((data.y.squeeze(1) == i), mask).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

        expected_l = 0.5 * rsv * len(index)
        l = math.floor(expected_l)
        l = l + (1 if (expected_l - l) >= random.uniform(0, 1) else 0)
        new_lens.append(l)

    index = torch.cat([arr[:new_lens[i]] for i, arr in enumerate(indices)], dim=0)

    num_to_select = round(rsv * len(idx)) - np.sum(new_lens)
    existing = set(index.cpu().numpy().tolist())
    to_select = []
    for i in ranks:
        if i not in existing:
            to_select.append(i)
            num_to_select -= 1
            if num_to_select <= 0:
                break
    index = torch.cat([index, torch.Tensor(to_select).long()], dim=0)

    return index
