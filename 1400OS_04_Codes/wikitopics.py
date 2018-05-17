from __future__ import print_function
import numpy as np
import logging, gensim

# LDA: Latent Dirichlet Allocation    潜在狄利克雷分配
# HDP: Hierarchical Dirichlet Process 层次狄利克雷过程
# TODO 涉及14G的语料, 整体运行估计要10+小时, 目前没调试
# 1. 下载语料(数十小时): wget http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
# 2. 构建索引(数小时): python -m gensim.scripts.make_wiki enwiki-latest-pages-articles.xml.bz2 wiki_en_output
# 然后再运行后续代码

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
id2word = gensim.corpora.Dictionary.load_from_text('data/wiki_en_output_wordids.txt')
mm = gensim.corpora.MmCorpus('data/wiki_en_output_tfidf.mm')

# 3. 建模型(数小时) 潜在狄利克雷分配 (LDA: Latent Dirichlet Allocation)
model = gensim.models.ldamodel.LdaModel(
    corpus=mm,
    id2word=id2word,
    num_topics=100,
    update_every=1,
    chunksize=10000,
    passes=1)  # 数小时
model.save('wiki_lda.pkl')

# 层次狄利克雷过程 (HDP: Hierarchical Dirichlet Process)
# hdp = gensim.models.hdpmodel.HdpModel(mm, id2word)  # 不需要指定主题个数，训练时间更长，之后使用过程跟LDA一样

# 4. 模型应用
# model = gensim.models.ldamodel.LdaModel.load('wiki_lda.pkl')  # 模型保存后，后续使用可以直接加载，省去 1, 2, 3步

topics = [model[doc] for doc in mm]
lens = np.array([len(t) for t in topics])
print(np.mean(lens <= 10))  # 0.932382190219 主题小于等于10个的文档数大约为 93%
print(np.mean(lens))  # 6.55842326445 平均每个文档有6.5个主题

# 收集主题使用情况统计信息
counts = np.zeros(100)
for doc_top in topics:
    for ti, _ in doc_top:
        counts[ti] += 1

words = model.show_topic(counts.argmax(), 64)  # 高频主题
print(words)
print()
print()
print()
words = model.show_topic(counts.argmin(), 64)  # 低频主题
print(words)
print()
print()
print()
