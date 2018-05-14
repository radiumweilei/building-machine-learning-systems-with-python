from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

# 鸢尾花数据集4个维度数据6种图形可视化效果
data = load_iris()
features = data['data']  # sepal length 花萼长度, sepal width 花萼宽度, petal length 花瓣长度, petal width 花瓣宽度
feature_names = data['feature_names']
target = data['target']

pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # 4类数据任取两个作图，总共6种类型的图
for i, (p0, p1) in enumerate(pairs):
    plt.subplot(2, 3, i + 1)
    for t, marker, c in zip(range(3), ">ox", "rgb"):
        plt.scatter(features[target == t, p0], features[target == t, p1], marker=marker, c=c)
    plt.xlabel(feature_names[p0])
    plt.ylabel(feature_names[p1])
    plt.xticks([])
    plt.yticks([])
plt.savefig('../1400_02_01.png')
