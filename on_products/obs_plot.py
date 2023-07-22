import os
import argparse
import re

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def load_random(fold, xs, ys, lgs):
    #rsvs = ['1.0', '0.95', '0.9', '0.85', '0.8', '0.75', '0.7', '0.65', '0.6', '0.55', '0.5', '0.45', '0.4', '0.35', '0.3', '0.25', '0.2', '0.15', '0.1', '0.09', '0.08', '0.07', '0.0625', '0.06', '0.0575', '0.055', '0.0525', '0.05', '0.0475', '0.045', '0.0425', '0.04', '0.0375', '0.035', '0.0325', '0.03']
    #rsvs = ['1.0', '0.95', '0.9', '0.85', '0.8', '0.75', '0.7', '0.65', '0.6', '0.55', '0.5', '0.45', '0.4', '0.35', '0.3', '0.25', '0.2', '0.15']
    rsvs = ['1.0', '0.9', '0.8', '0.7', '0.6', '0.5', '0.4']

    accs = []
    for para in rsvs:
        #fn = os.path.join(fold, para+".out")
        fn = os.path.join(fold, "{}_{}.out".format(para, para))
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


def load_al(method, xs, ys, lgs, min_rsv, min_alpha):
    results = dict()
    #alphas = []
    #accs = []
    fold = os.path.join("logs", method)
    for fn in os.listdir(fold):
        if fn.endswith(".out"):
            params = re.findall("\d+\.\d+", fn)
            if len(params) != 2:
                continue
            rsv = float(params[0])
            if rsv < min_rsv:
                continue
            alpha = float(params[1])
            if alpha < min_alpha:
                continue
            if alpha not in results:
                results[alpha] = []
            executed = False
            with open(os.path.join(fold, fn), 'r') as ips:
                for line in ips:
                    mobj = re.findall("\d+\.\d+", line)
                    if len(mobj) == 2:
                        acc = float(mobj[0])
                        executed = True
            if executed:
                results[alpha].append((rsv, acc))
    for key in results:
        results[key] = sorted(results[key], key=lambda tp:tp[0])
        
    for key in sorted(results.keys()):
        lgs.append("{} {}%".format(method, int(key*100)))
        xs.append(np.asarray([tp[0] for tp in results[key]]))
        ys.append(np.asarray([tp[1] for tp in results[key]]))


def main():
    parser = argparse.ArgumentParser(description='Plot Results')
    parser.add_argument('--fold', type=str, default='logs')
    parser.add_argument('--al', type=str, default='')
    parser.add_argument('--min_rsv', type=float, default=0.0)
    parser.add_argument('--min_alpha', type=float, default=0.0)
    args = parser.parse_args()
    print(args)

    xs, ys, lgs = [], [], []
    load_random(os.path.join(args.fold, "random"), xs, ys, lgs)
    #if args.al:
    #    load_al(args.al, xs, ys, lgs, args.min_rsv, args.min_alpha)

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
    plt.savefig("obs_{}.pdf".format(args.al) if args.al else "obs_random.pdf", transparent='True')


if __name__ == "__main__":
    main()
