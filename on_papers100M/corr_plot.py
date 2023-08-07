import os
import argparse

import numpy as np
from scipy import stats


def load_al_rank(name):
    res = []
    with open(os.path.join(name, 'train.tsv'), 'r') as ips:
        for line in ips:
            idx, val = line.strip().split('\t')
            res.append((int(idx), float(val)))
    res = sorted(res, key=lambda x: x[0])
    return np.asarray([tp[1] for tp in res])


def main():
    parser = argparse.ArgumentParser(description='Plot Results')
    args = parser.parse_args()
    print(args)

    als = []
    for al in ['random', 'ours', 'ours1']:
        vals = load_al_rank(al)
        if al == 'ddd':
            vals = np.max(vals) - vals
        als.append(vals)

    corr_matrix = []
    for i in range(len(als)):
        row = []
        for j in range(len(als)):
            if i == j:
                row.append(1.0)
                continue
            statistic, pval = stats.spearmanr(als[i], als[j])
            row.append(statistic)
        corr_matrix.append(row)
        print(row)


if __name__ == "__main__":
    main()
