from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, silhouette_score
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

# 解决UserWarning
import os
os.environ["OMP_NUM_THREADS"] = '1'

# 加载Wine数据集
wine = load_wine()
X = wine.data
y = wine.target

# 降维前的数据
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
acc = accuracy_score(y, labels)
ami = adjusted_mutual_info_score(y, labels)
silhouette = silhouette_score(X, labels)

# 打印结果
print("Metrics:")
print("ACC:", acc)
print("AMI:", ami)
print("Silhouette:", silhouette)

# 可视化聚类结果
# KMeans降维前聚类结果可视化
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=60)
plt.title('KMeans Clustering')

plt.show()