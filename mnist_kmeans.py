# 解决UserWarning
import os
os.environ["OMP_NUM_THREADS"] = '4'

# 忽略FutureWarning
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=870)

import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import silhouette_score

def read_idx(filename):
    """
    读取 IDX 文件格式的函数
    """
    with open(filename, 'rb') as f:
        # 读取魔数，数据类型，维度信息
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        # 计算数据总数
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        # 读取数据
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# print("test set shape:", test_images.shape)
# print("test set labels shape:", test_labels.shape)

# 读取训练集图像和标签
train_images = read_idx('./data/mnist/train-images.idx3-ubyte')
train_labels = read_idx('./data/mnist/train-labels.idx1-ubyte')

# 选取每个类别随机的 100 张图片
new_images = []
new_labels = []
for i in range(10):
    indices = np.where(train_labels == i)[0]
    selected_indices = np.random.choice(indices, size=100, replace=False)
    new_images.append(train_images[selected_indices])
    new_labels.append(np.ones(100, dtype=np.uint8) * i)

# 将选取的图像和标签合并为一个数组
new_images = np.concatenate(new_images, axis=0)
new_labels = np.concatenate(new_labels, axis=0)

# 打乱训练集的顺序
shuffle_indices = np.random.permutation(len(new_images))
new_images = new_images[shuffle_indices]
new_labels = new_labels[shuffle_indices]

# # 选取前 5 张图像可视化
# for i in range(10):
#     plt.subplot(1, 10, i+1)
#     plt.imshow(new_images[i], cmap='gray')
#     plt.axis('off')
#     plt.title(str(new_labels[i]))
# plt.show()

# 将图像拉伸为一个向量
new_images = new_images.reshape((1000, 784))

# 将每个数据归一化处理
new_images = new_images / 255.0

# # 打印第一张图像归一化后的像素值
# print(new_images[0])

print("New set shape:", new_images.shape)
print("New set labels shape:", new_labels.shape)

tsne = TSNE(n_components=2, random_state=123)
new_images_tsne = tsne.fit_transform(new_images)


# 使用KMeans算法进行聚类
kmeans = KMeans(n_clusters=10, random_state=123, n_init=10)
kmeans.fit(new_images)

# 将聚类结果转换为标签
pred_labels = np.zeros_like(new_labels)
for i in range(10):
    mask = (kmeans.labels_ == i)    # 这个掩码数组将用于选择属于当前簇的样本
    pred_labels[mask] = np.bincount(new_labels[mask]).argmax()    # pred_labels数组中存储了每个样本聚类后的标签

print("原始784维:")
# 计算聚类准确率
acc = accuracy_score(new_labels, pred_labels)
print("     Clustering accuracy:", acc)

# 计算带矫正的AMI
ami = adjusted_mutual_info_score(new_labels, pred_labels)
print("     Adjusted Mutual Information:", ami)

# 计算带矫正的silhouette score
silhouette = silhouette_score(new_images_tsne, pred_labels)
print("     Silhouette score:", silhouette)



'''
先用TSNE降维，可以减少噪音和冗余，提高聚类的准确性
再用KMeans聚类
'''
# 使用TSNE降维
kmeans_tsne = KMeans(n_clusters=10, random_state=123, n_init=10)
kmeans_tsne.fit(new_images_tsne)

# 将聚类结果转换为标签
pred_labels_tsne = np.zeros_like(new_labels)
for i in range(10):
    mask = (kmeans_tsne.labels_ == i)    # 这个掩码数组将用于选择属于当前簇的样本
    pred_labels_tsne[mask] = np.bincount(new_labels[mask]).argmax()    # pred_labels_tsne数组中存储了每个样本聚类后的标签

print("先用TSNE降维:")
# 计算聚类准确率
acc = accuracy_score(new_labels, pred_labels_tsne)
print("     Clustering accuracy:", acc)

# 计算带矫正的AMI
ami = adjusted_mutual_info_score(new_labels, pred_labels_tsne)
print("     Adjusted Mutual Information:", ami)

# 计算带矫正的silhouette score
silhouette = silhouette_score(new_images_tsne, pred_labels_tsne)
print("     Silhouette score:", silhouette)


'''
对聚类结果进行可视化
'''
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为中文黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 使用散点图可视化聚类结果
plt.figure(figsize=(10, 10))
colors = plt.cm.get_cmap('Paired', 10) # 获取Paired颜色，共10种
for i in range(10):
    mask = (pred_labels_tsne == i)
    plt.scatter(new_images_tsne[mask, 0], new_images_tsne[mask, 1], color=colors(i), label='Cluster %d' % i)
plt.legend()
ax = plt.gca()
ax.axis('off')
plt.suptitle("KMeans聚类结果可视化")

# 若无则创建result文件夹
import os
if not os.path.exists('./result'):
    os.mkdir('./result')

# 保存聚类结果图片
plt.savefig('./result/mnist_kmeans_cluster.png')

plt.show()

