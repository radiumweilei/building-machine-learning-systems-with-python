from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

# 鸢尾花数据集4个维度数据6种图形可视化效果
#
# 数据源: <python_home>/lib/python3.5/site-packages/sklearn/datasets/data/iris.csv
# 数据格式: 花萼长度,花萼宽度,花瓣长度,花瓣宽度,鸢尾花的类别
# 数据描述: 第一行为简要说明, 实际数据从第二行开始, 第151行结束, 总共150条数据, 分三类各50条
# 鸢尾花的类别: 0 - Iris Setosa 山鸢尾花, 1 - Iris Versicolor 变色鸢尾花, 2 - Iris Virginica 维吉尼亚鸢尾花
#
# Data Set Characteristics:
#     :Number of Instances: 150 (50 in each of three classes)
#     :Number of Attributes: 4 numeric, predictive attributes and the class
#     :Attribute Information:
#         - sepal length in cm
#         - sepal width in cm
#         - petal length in cm
#         - petal width in cm
#         - class:
#                 - 0 Iris-Setosa
#                 - 1 Iris-Versicolour
#                 - 2 Iris-Virginica
#     :Summary Statistics:
#
#     ============== ==== ==== ======= ===== ====================
#                     Min  Max   Mean    SD   Class Correlation
#     ============== ==== ==== ======= ===== ====================
#     sepal length:   4.3  7.9   5.84   0.83    0.7826
#     sepal width:    2.0  4.4   3.05   0.43   -0.4194
#     petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
#     petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
#     ============== ==== ==== ======= ===== ====================
#
#     :Missing Attribute Values: None
#     :Class Distribution: 33.3% for each of 3 classes.

data = load_iris()
features = data['data']  # sepal length 花萼长度, sepal width 花萼宽度, petal length 花瓣长度, petal width 花瓣宽度
feature_names = data['feature_names']  # 4个数据维度特征名
target = data['target']  # 分类向量, 长度150, 0-setosa, 1-versicolor, 2-virginical

pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # 4类数据任取两个作图，总共6种类型的图
for i, (p0, p1) in enumerate(pairs):
    plt.subplot(2, 3, i + 1)  # 绘制2*3=6个图, 当前是第 i+1 个图, 下标从1开始行序优先
    # 每种花用不同的颜色绘制, 3组值: 0 >(三角形) r, 1 o(圆圈) g, 2 x(X号) b; 用 t 来筛选 0,1,2 三类花的数据
    for t, marker, c in zip(range(3), ">ox", "rgb"):
        # marker 可用值参考 matplotlib.markers
        plt.scatter(features[target == t, p0], features[target == t, p1], marker=marker, c=c)
    plt.xlabel(feature_names[p0])
    plt.ylabel(feature_names[p1])
    plt.xticks([])
    plt.yticks([])
plt.savefig('../1400_02_01.png')
