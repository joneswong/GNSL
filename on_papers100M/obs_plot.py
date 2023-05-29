import os
import argparse
import re

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def load_random(fold, xs, ys, lgs):
    rsvs = ['1.0', '0.9', '0.8', '0.7', '0.6', '0.5', '0.475', '0.45', '0.425', '0.4', '0.375', '0.35', '0.325', '0.3', '0.275', '0.25', '0.225', '0.2', '0.175', '0.15', '0.125', '0.1']

    accs = []
    for para in rsvs:
        fn = os.path.join(fold, para+".out")
        with open(fn, 'r') as ips:
            for line in ips:
                mobj = re.findall("\d+\.\d+", line)
                if len(mobj) == 2:
                    acc = float(mobj[0])
            accs.append(acc)
    x, y = np.asarray([float(elem) for elem in rsvs]), np.asarray(accs)
    xs.append(x)
    ys.append(y)
    lgs.append('random')


def main():
    parser = argparse.ArgumentParser(description='Plot Results')
    parser.add_argument('--fold', type=str, default='logs')
    parser.add_argument('--al', type=str, default='random')
    parser.add_argument('--min_rsv', type=float, default=0.0)
    parser.add_argument('--min_alpha', type=float, default=0.0)
    args = parser.parse_args()
    print(args)

    xs, ys, lgs = [], [], []
    load_random(os.path.join(args.fold, args.al), xs, ys, lgs)
    #load_al(args.al, xs, ys, lgs, args.min_rsv, args.min_alpha)

    #1207179 125265
    V = 1332384
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
    plt.savefig("obs_{}_on_papers100M.pdf".format(args.al), transparent='True')


if __name__ == "__main__":
    main()
