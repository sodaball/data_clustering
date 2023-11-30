from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
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

# 使用PCA进行降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 使用UMAP进行降维
umap = UMAP(n_components=2, random_state=42)
X_umap = umap.fit_transform(X)

'''
KMeans聚类
'''
# PCA降维后的数据
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_pca = kmeans_pca.fit_predict(X_pca)
acc_pca = accuracy_score(y, labels_pca)
ami_pca = adjusted_mutual_info_score(y, labels_pca)
silhouette_pca = silhouette_score(X_pca, labels_pca)

# t-SNE降维后的数据
kmeans_tsne = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_tsne = kmeans_tsne.fit_predict(X_tsne)
acc_tsne = accuracy_score(y, labels_tsne)
ami_tsne = adjusted_mutual_info_score(y, labels_tsne)
silhouette_tsne = silhouette_score(X_tsne, labels_tsne)

# UMAP降维后的数据
kmeans_umap = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_umap = kmeans_umap.fit_predict(X_umap)
acc_umap = accuracy_score(y, labels_umap)
ami_umap = adjusted_mutual_info_score(y, labels_umap)
silhouette_umap = silhouette_score(X_umap, labels_umap)


'''
打印结果
'''
print("先用PCA降维:")
print("accuracy:", acc_pca)
print("AMI:", ami_pca)
print("Silhouette:", silhouette_pca)

print("先用TSNE降维:")
print("accuracy:", acc_tsne)
print("AMI:", ami_tsne)
print("Silhouette:", silhouette_tsne)

print("先用UMAP降维:")
print("accuracy:", acc_umap)
print("AMI:", ami_umap)
print("Silhouette:", silhouette_umap)

'''
可视化聚类结果
'''

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为中文黑体

# PCA聚类结果可视化
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pca, cmap='viridis', s=60)
plt.title('PCA + KMeans聚类结果可视化')
ax = plt.gca()
ax.axis('off')
# 若无则创建result文件夹
import os
if not os.path.exists('./result'):
    os.mkdir('./result')
# 保存聚类结果图片
plt.savefig('./result/wine_kmeans_pca_cluster.png')
plt.show()

# TSNE聚类结果可视化
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_tsne, cmap='viridis', s=60)
plt.title('TSNE + KMeans聚类结果可视化')
ax = plt.gca()
ax.axis('off')
plt.savefig('./result/wine_kmeans_tsne_cluster.png')
plt.show()

# UMAP聚类结果可视化
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels_umap, cmap='viridis', s=60)
plt.title('UMAP + KMeans聚类结果可视化')
ax = plt.gca()
ax.axis('off')
plt.savefig('./result/wine_kmeans_umap_cluster.png')
plt.show()