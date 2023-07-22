with open("infl-max/new_train.tsv", 'w') as ops:
    with open("infl-max/train.tsv", 'r') as ips:
        for line in ips:
            tp = line.strip().split('\t')
            ops.write("{}\t{}\n".format(tp[0][1:-1], tp[1]))
