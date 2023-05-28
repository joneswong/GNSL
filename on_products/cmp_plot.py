import os
import argparse
import re

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def load_random(fold):
    rsvs = ['1.0', '0.95', '0.9', '0.85', '0.8', '0.75', '0.7', '0.65', '0.6', '0.55', '0.5', '0.45', '0.4', '0.35', '0.3', '0.25', '0.2', '0.15', '0.1', '0.09', '0.08', '0.07', '0.0625', '0.06', '0.0575', '0.055', '0.0525', '0.05', '0.0475', '0.045', '0.0425', '0.04', '0.0375', '0.035', '0.0325', '0.03']
    accs = []
    for para in rsvs:
        fn = os.path.join(fold, para+".out")
        with open(fn, 'r') as ips:
            for line in ips:
                mobj = re.findall("\d+\.\d+", line)
                if len(mobj) == 2:
                    acc = float(mobj[0])
            print(acc)
            accs.append(acc)
    return np.asarray([float(elem) for elem in rsvs]), np.asarray(accs)


def load_mem(fold):
    alphas = []
    accs = []
    for fn in os.listdir(fold):
        if fn.endswith(".out"):
            alpha = float(re.findall("\d+\.\d+", fn)[1])
            alphas.append(alpha)

            with open(os.path.join(fold, fn), 'r') as ips:
                for line in ips:
                    mobj = re.findall("\d+\.\d+", line)
                    if len(mobj) == 2:
                        acc = float(mobj[0])
                print(acc)
                accs.append(acc)
    results = sorted([(x, y) for x, y in zip(alphas, accs)], key=lambda tp:tp[0])
    xs = [tp[0] for tp in results]
    ys = [tp[1] for tp in results]
    return np.asarray(xs), np.asarray(ys)


def main():
    parser = argparse.ArgumentParser(description='Plot Results')
    parser.add_argument('--fold', type=str, default='logs')
    args = parser.parse_args()
    print(args)

    xs, ys, lgs = [], [], []
    tp = load_random(os.path.join(args.fold, "random"))
    xs.append(tp[0])
    ys.append(tp[1])
    lgs.append('Random')

    tp = load_mem(os.path.join(args.fold, "mem"))
    xs.append(tp[0])
    ys.append(tp[1])
    lgs.append('Memorization')

    V = 235938
    for i in range(len(xs)):
        xs[i] = V * xs[i]
        ys[i] = 100.0 - ys[i]

    # plt.axvline(np.sqrt(N)/2)
    with sns.color_palette('viridis_r', len(xs)):
        plt.figure(figsize=(6, 5))
        for i in range(len(xs)):
            plt.scatter(xs[i], ys[i], label=lgs[i])
            plt.plot(xs[i], ys[i])
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Error')
    #plt.xlabel(r'$\alpha$')
    plt.xlabel('Training examples')
    plt.legend();
    #plt.xticks([1,2,3],[1,2,3])
    #plt.yticks([2,10,20,50],[2,10,20,50])
    #plt.ylim([2,50])
    plt.grid(True,which='both',alpha=0.2)
    sns.despine()
    plt.savefig("plot.pdf", transparent='True')


if __name__ == "__main__":
    main()
