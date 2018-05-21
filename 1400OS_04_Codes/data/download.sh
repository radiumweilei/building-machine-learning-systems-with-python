#!/bin/sh
wget http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2  # wiki数据集，大得惊人，15GB大小，要命
#wget http://www.cs.princeton.edu/~blei/lda-c/ap.tgz  # 该文件找不着了，新的地址如下
wget http://www.cs.columbia.edu/~blei/lda-c/ap.tgz  # 为避免以后找不着该数据集，已经将该数据集提交到git上了
tar xzf ap.tgz

# enwiki-latest-pages-articles.xml.bz2 已经存储在移动硬盘 /code/python/data/building-machine-learning-systems-with-python/1400OS_05_Codes/ 目录下