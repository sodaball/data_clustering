import numpy as np
import matplotlib.pyplot as plt

# K-Means算法
# X: 数据集
# K: 聚类数
# max_iters: 最大迭代次数
def kmeans(X, K, max_iters=100):
    centroids = X[np.random.choice(range(X.shape[0]), size=K, replace=False), :]
    for _ in range(max_iters):
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        new_centroids = np.array([X[labels==k].mean(axis=0) for k in range(K)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# 从本地加载数据集iris.data
X = np.loadtxt('./data/iris/iris.data', delimiter=',', usecols=(0, 1, 2, 3)) # 前四列为特征，第五列为标签，KMenas只使用前四列特征
# print(X.shape)
# print(X)

# 聚类
labels, centroids = kmeans(X, 3)    # 返回聚类标签和聚类中心

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')   # 绘制散点图
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5);  # 绘制聚类中心

# 若无则创建result文件夹
import os
if not os.path.exists('./result'):
    os.mkdir('./result')

# 保存聚类结果图片
plt.savefig('./result/iris_cluster.png')

# 显示聚类结果图片
plt.show()

