# case 07 Pcn55 美国联合通讯社AP数据集, 数据源: ./data

from __future__ import print_function
from gensim import corpora, models
# from mpltools import style  已集成到 matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
from os import path

style.use('ggplot')

if not path.exists('./data/ap/ap.dat'):
    print('Error: Expected data to be present at data/ap/')

corpus = corpora.BleiCorpus('./data/ap/ap.dat', './data/ap/vocab.txt')  # http://www.cs.columbia.edu/~blei/lda-c/ap.tgz
model = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word, alpha=None)

for ti in range(84):
    words = model.show_topic(ti, 64)
    tf = sum(f for w, f in words)
    print('\n'.join('{}:{}'.format(w, int(1000. * f / tf)) for w, f in words))
    print()
    print()
    print()

thetas = [model[c] for c in corpus]
plt.hist([len(t) for t in thetas], np.arange(42))
plt.ylabel('Nr of documents')
plt.xlabel('Nr of topics')
plt.savefig('./1400OS_04_01+.png')

model1 = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word, alpha=1.)
thetas1 = [model1[c] for c in corpus]

# model8 = models.ldamodel.LdaModel(corpus, num_topics=100, id2word=corpus.id2word, alpha=1.e-8)
# thetas8 = [model8[c] for c in corpus]
plt.clf()
plt.hist([[len(t) for t in thetas], [len(t) for t in thetas1]], np.arange(42))
plt.ylabel('Nr of documents')
plt.xlabel('Nr of topics')
plt.text(9, 223, r'default alpha')
plt.text(26, 156, 'alpha=1.0')
plt.savefig('./1400OS_04_02+.png')
