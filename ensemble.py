import os
import math
import random
import argparse


def load_rank_list(al):
    ranks = []
    with open(os.path.join(al, "train.tsv"), 'r') as ips:
        for line in ips:
            idx, _ = line.strip().split('\t')
            idx = int(idx)
            ranks.append(idx)
    return ranks


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products')
    parser.add_argument('--dataset', type=str, default='products')
    parser.add_argument('--individuals', type=str, default='mem,dist-greedy')
    parser.add_argument('--weights', type=str, default='0.5,0.5')
    args = parser.parse_args()
    print(args)

    fold = "on_{}".format(args.dataset)
    vals = dict()
    for wt, al in zip(args.weights.split(','), args.individuals.split(',')):
        rank = load_rank_list(os.path.join(fold, al))
        weight = float(wt)
        for i, idx in enumerate(rank):
            if idx in vals:
                vals[idx] += weight * (1.0 - i / float(len(rank)))
            else:
                vals[idx] = weight * (1.0 - i / float(len(rank)))
    rank = sorted([(k, v) for k, v in vals.items()], key=lambda x:x[1], reverse=True)

    with open("train.tsv", 'w') as ops:
        for i in range(len(rank)):
            idx, val = rank[i]
            ops.write("{}\t{}\n".format(idx, val))


if __name__=='__main__':
    main()
