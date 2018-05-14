import milksets.iris
import milksets.seeds


def save_as_tsv(fname, module):
    features, labels = module.load()
    nlabels = [module.label_names[ell] for ell in labels]
    with open(fname, 'w') as ofile:
        print(fname)
        for f, n in zip(features, nlabels):
            # print >> ofile, "\t".join(map(str, f) + [n])  # python 2
            print("\t".join(list(map(str, f)) + [n]), file=ofile)  # python 3


save_as_tsv('iris.tsv', milksets.iris)  # 奇怪为何会写到 extra 目录下? 这里添加 ../ 存回上一级目录，方便 test_load.py 执行
save_as_tsv('seeds.tsv', milksets.seeds)
