from matplotlib import pyplot as plt
from load_ml100k import load

data = load()
data = data.toarray()
plt.gray()
plt.imshow(data[:200, :200], interpolation='nearest')
plt.xlabel('User ID')
plt.ylabel('Film ID')
plt.savefig('../charts/1400_08_03.png')
