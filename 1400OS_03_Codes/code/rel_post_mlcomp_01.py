# case 06 Pcn49 20newsgroup数据集聚类, 数据源: ../../data

import sklearn.datasets
import scipy as sp

new_post = \
    """Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks.
"""

# 下载 wget http://mlcomp.org/download/dataset-379-20news-18828_SOPFP.zip， 如果需要注册请先注册，解压出目录 379
# 将目录 379 改成 20news-18828, 将绝对路径设到MLCOMP_DIR变量中
# 参考 sklearn.datasets.load_mlcomp 的注释:
#   since the http://mlcomp.org/ website will shut down
#   in March 2017, the load_mlcomp function was deprecated
#   in version 0.19 and will be removed in 0.21.
# 这里将已解决好的数据集提交到git中方便后续的学习研究, 直接从git中pull出来然后解压即可正常使用
MLCOMP_DIR = "../../data"
groups = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.ma c.hardware', 'comp.windows.x', 'sci.space']
dataset = sklearn.datasets.load_mlcomp("20news-18828", "train", mlcomp_root=MLCOMP_DIR, categories=groups)
print("Number of posts:", len(dataset.filenames))

labels = dataset.target
num_clusters = 50  # sp.unique(labels).shape[0]

import nltk.stem

english_stemmer = nltk.stem.SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,
                                    # max_features=1000,
                                    stop_words='english',
                                    # charset_error='ignore' # 该参数已被废弃，新的参数为 decode_error
                                    decode_error='ignore'
                                    )
vectorized = vectorizer.fit_transform(dataset.data)
num_samples, num_features = vectorized.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))

from sklearn.cluster import KMeans

km = KMeans(n_clusters=num_clusters, init='k-means++', n_init=1,
            verbose=1)

clustered = km.fit(vectorized)

from sklearn import metrics

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand Index: %0.3f" %
      metrics.adjusted_rand_score(labels, km.labels_))
print("Adjusted Mutual Information: %0.3f" %
      metrics.adjusted_mutual_info_score(labels, km.labels_))
print(("Silhouette Coefficient: %0.3f" %
       metrics.silhouette_score(vectorized, labels, sample_size=1000)))

new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]

similar_indices = (km.labels_ == new_post_label).nonzero()[0]

similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
    similar.append((dist, dataset.data[i]))

similar = sorted(similar)

# 调试器，没毛用，注释掉
# import pdb
# pdb.set_trace()

show_at_1 = similar[0]
show_at_2 = similar[int(len(similar) / 2)]
show_at_3 = similar[-1]

print(show_at_1)
print(show_at_2)
print(show_at_3)
