# case 01 Pcn10 假想的互联网公司访问量预测模型, 多项式拟合

import os
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 案例1
# 互联网公司MLAAS，通过HTTP向用户推销机器学习算法服务，提供优质服务需要更好的基础设施。
# 基础设施多了会浪费钱，少了服务质量就不能保证会导致赔钱。
# 问题：何时会达到基础设施的服务极限，目前估计的极限是: 100000请求/小时
# 源数据：../data/web_traffic.tsv: 小时, 访问量

# all examples will have three classes in this file
colors = ['g', 'k', 'b', 'm', 'r']
linestyles = ['-', '-.', '--', ':', '-']


# 公共绘图函数
def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):
    plt.clf()
    plt.scatter(x, y, s=10)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks([w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)])

    if models:
        if mx is None:
            mx = sp.linspace(0, x[-1], 1000)
        for model, style, color in zip(models, linestyles, colors):
            # print "Model:",model
            # print "Coeffs:",model.coeffs
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    plt.savefig(fname)


# 误差计算函数: 对于一个训练好的模型f，按照如下公式计算其误差，f(x)表示使用模型后的结果, sp.sum((f(x) - y) ** 2) 为所有误差平方和
# f(x): scipy提供的向量化函数
def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)


# 0. 加载数据, 初始化向量
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
data = sp.genfromtxt(os.path.join(data_dir, "web_traffic.tsv"), delimiter="\t")
print(data[:10])
x = data[:, 0]  # 小时信息
y = data[:, 1]  # 某个小时的Web访问数
print("Number of invalid entries:", sp.sum(sp.isnan(y)))  # 无效数据

# 1. 去掉无效数据后，绘制当前数据的图形
x = x[~sp.isnan(y)]  # 只取合法数据
y = y[~sp.isnan(y)]  # 只取合法数据
plot_models(x, y, None, os.path.join("..", "1400_01_01.png"))

# 2. 创建模型，做1阶、2阶、3阶、10阶、100阶拟合并绘图
fp1, res, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)  # fp1 为模型参数
print("Model parameters: %s" % fp1)
print("Error of the model:", res)
f1 = sp.poly1d(fp1)
f2 = sp.poly1d(sp.polyfit(x, y, 2))
f3 = sp.poly1d(sp.polyfit(x, y, 3))
f10 = sp.poly1d(sp.polyfit(x, y, 10))
f100 = sp.poly1d(sp.polyfit(x, y, 100))
plot_models(x, y, [f1], os.path.join("..", "1400_01_02.png"))
plot_models(x, y, [f1, f2], os.path.join("..", "1400_01_03.png"))
plot_models(x, y, [f1, f2, f3, f10, f100], os.path.join("..", "1400_01_04.png"))

# 3. 依据已有数据在3.5周左右出现大的拐点进行调整并设计模型
inflection = int(3.5 * 7 * 24)
xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = y[inflection:]
fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))
plot_models(x, y, [fa, fb], os.path.join("..", "1400_01_05.png"))  # 相比其他复查模型，最后一周更符合该直线模型，更符合未来数据

print("Errors for the complete data set:")
for f in [f1, f2, f3, f10, f100]:
    print("Error d=%i: %f" % (f.order, error(f, x, y)))

print("Errors for only the time before inflection point")
for f in [f1, f2, f3, f10, f100]:
    print("Error d=%i: %f" % (f.order, error(f, xa, ya)))

print("Errors for only the time after inflection point")
for f in [f1, f2, f3, f10, f100]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))

print("Error inflection=%f" % (error(fa, xa, ya) + error(fb, xb, yb)))

# 4. 预测 extrapolating into the future, 对于10阶、100阶两种情况从图中看出及预测效果非常差（过拟合导致）
plot_models(
    x, y, [f1, f2, f3, f10, f100], os.path.join("..", "1400_01_06.png"),
    mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)

# 5. 仅仅用拐点后的数据做模型训练
print("Trained only on data after inflection point")
fb1 = fb
fb2 = sp.poly1d(sp.polyfit(xb, yb, 2))
fb3 = sp.poly1d(sp.polyfit(xb, yb, 3))
fb10 = sp.poly1d(sp.polyfit(xb, yb, 10))
fb100 = sp.poly1d(sp.polyfit(xb, yb, 100))

print("Errors for only the time after inflection point")
for f in [fb1, fb2, fb3, fb10, fb100]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))

plot_models(
    x, y, [fb1, fb2, fb3, fb10, fb100], os.path.join("..", "1400_01_07.png"),
    mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)

# 6. 从给定数据中分离出训练数据和测试数据, separating training from testing data
frac = 0.3
split_idx = int(frac * len(xb))
shuffled = sp.random.permutation(list(range(len(xb))))
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])
fbt1 = sp.poly1d(sp.polyfit(xb[train], yb[train], 1))
fbt2 = sp.poly1d(sp.polyfit(xb[train], yb[train], 2))
fbt3 = sp.poly1d(sp.polyfit(xb[train], yb[train], 3))
fbt10 = sp.poly1d(sp.polyfit(xb[train], yb[train], 10))
fbt100 = sp.poly1d(sp.polyfit(xb[train], yb[train], 100))

print("Test errors for only the time after inflection point")
for f in [fbt1, fbt2, fbt3, fbt10, fbt100]:
    print("Error d=%i: %f" % (f.order, error(f, xb[test], yb[test])))

plot_models(
    x, y, [fbt1, fbt2, fbt3, fbt10, fbt100], os.path.join("..", "1400_01_08.png"),  # 该图具有很大随机性,  取决于shuffled
    mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)

# 7. 通过比较，fbt2的误差最小，选定fbt2作为我们的模型，对未来进行预测，看访问量到达100000会在什么时间点发生
print(fbt2)
print(fbt2 - 100000)
reached_max = fsolve(fbt2 - 100000, 800) / (7 * 24)
print("100,000 hits/hour expected at week %f" % reached_max[0])
