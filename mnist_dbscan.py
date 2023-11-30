import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.metrics import make_scorer, silhouette_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import silhouette_score
from tqdm import tqdm

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

# 将图像拉伸为一个向量
new_images = new_images.reshape((1000, 784))

# 将每个数据归一化处理
new_images = new_images / 255.0

print("New set shape:", new_images.shape)
print("New set labels shape:", new_labels.shape)

# 使用TSNE降维
tsne = TSNE(n_components=2, random_state=0)
new_images_tsne = tsne.fit_transform(new_images)


'''
画折线图，横坐标为eps，纵坐标为accuracy
每个min_samples画一条折线图，并将accuracy达到最大时的eps和min_samples保存起来
'''
# 定义超参数的范围
eps_values = np.linspace(2.00, 15.00, num=1300)
min_samples_values = range(2, 15)

best_accuracy = 0.0
best_eps = None
best_min_samples = None

for min_samples in tqdm(min_samples_values, desc='min_samples', colour = 'red'):
    eps_list = []
    accuracy_list = []
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(new_images)

        pred_labels = np.zeros_like(new_labels, dtype=np.int32)
        for i in range(10):
            mask = (dbscan.labels_ == i)
            if np.sum(mask) == 0:
                pred_labels[mask] = -1
            else:
                pred_labels[mask] = np.bincount(new_labels[mask]).argmax()

        accuracy = accuracy_score(new_labels, pred_labels)
        eps_list.append(eps)
        accuracy_list.append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_eps = eps
            best_min_samples = min_samples
            
    plt.plot(eps_list, accuracy_list, label=f"min_samples={min_samples}")

plt.title("Accuracy vs. Eps")
plt.xlabel("Eps")
plt.ylabel("Accuracy")
plt.legend()

# 若无则创建result文件夹
import os
if not os.path.exists('./result'):
    os.mkdir('./result')

# 保存聚类结果图片
plt.savefig('./result/mnist_dbscan_min_samples_eps.png')

plt.show()

print(f"Best accuracy: {best_accuracy:.2f}")
print(f"Best eps: {best_eps:.2f}")
print(f"Best min_samples: {best_min_samples}")

'''
降维之后找最佳超参数
'''
# 定义超参数的范围
eps_values = np.linspace(2.00, 15.00, num=1300)
min_samples_values = range(2, 15)

best_accuracy_tsne = 0.0
best_eps_tsne = None
best_min_samples_tsne = None

for min_samples in tqdm(min_samples_values, desc='min_samples', colour = 'green'):
    eps_list = []
    accuracy_list = []
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(new_images_tsne)

        pred_labels = np.zeros_like(new_labels, dtype=np.int32)
        for i in range(10):
            mask = (dbscan.labels_ == i)
            if np.sum(mask) == 0:
                pred_labels[mask] = -1
            else:
                pred_labels[mask] = np.bincount(new_labels[mask]).argmax()

        accuracy = accuracy_score(new_labels, pred_labels)
        eps_list.append(eps)
        accuracy_list.append(accuracy)
        
        if accuracy > best_accuracy_tsne:
            best_accuracy_tsne = accuracy
            best_eps_tsne = eps
            best_min_samples_tsne = min_samples
            
    plt.plot(eps_list, accuracy_list, label=f"min_samples={min_samples}")

plt.title("Accuracy vs. Eps (TSNE)")
plt.xlabel("Eps_tsne")
plt.ylabel("Accuracy_tsne")
plt.legend()

# 若无则创建result文件夹
import os
if not os.path.exists('./result'):
    os.mkdir('./result')

# 保存聚类结果图片
plt.savefig('./result/mnist_dbscan_min_samples_eps_tsne.png')

plt.show()

print(f"Best accuracy_tsne: {best_accuracy_tsne:.2f}")
print(f"Best eps_tsne: {best_eps_tsne:.2f}")
print(f"Best min_samples_tsne: {best_min_samples_tsne}")


print(f"Best eps: {best_eps:.2f}")
print(f"Best min_samples: {best_min_samples}")

print(f"Best eps_tsne: {best_eps_tsne:.2f}")
print(f"Best min_samples_tsne: {best_min_samples_tsne}")


'''
利用上一步得到的最佳超参数作为dbscan的超参数
计算聚类准确度、AMI和silhouette score
'''
# 创建一个具有定义超参数的DBSCAN对象
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan.fit(new_images)

# 将聚类结果转换为标签
pred_labels = np.zeros_like(new_labels, dtype=np.int32)
for i in range(10):
    mask = (dbscan.labels_ == i)
    if np.sum(mask) == 0:
        pred_labels[mask] = -1  # 如果为空，则将对应的pred_labels赋值为-1
    else:
        pred_labels[mask] = np.bincount(new_labels[mask]).argmax()  # 否则将pred_labels赋值为众数

print("原始784维:")     
# 计算聚类的准确度分数
accuracy = accuracy_score(new_labels, pred_labels)
print("     Clustering accuracy：", accuracy)

# 计算带矫正的AMI
ami = adjusted_mutual_info_score(new_labels, pred_labels)
print("     Adjusted Mutual Information:", ami)

# 计算带矫正的silhouette score
silhouette = silhouette_score(new_images, pred_labels)
print("     Silhouette score:", silhouette)


'''
先用TSNE降维，可以减少噪音和冗余，提高聚类的准确性
再用DBSCAN聚类
'''
# 使用TSNE降维
dbscan_tsne = DBSCAN(eps=best_eps_tsne, min_samples=best_min_samples_tsne)
dbscan_tsne.fit(new_images_tsne)

# 将聚类结果转换为标签
pred_labels_tsne = np.zeros_like(new_labels)
for i in range(10):
    mask = (dbscan_tsne.labels_ == i)
    if np.sum(mask) == 0:
        pred_labels_tsne[mask] = -1  # 如果为空，则将对应的pred_labels赋值为-1
    else:
        pred_labels_tsne[mask] = np.bincount(new_labels[mask]).argmax()  # 否则将pred_labels赋值为众数

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
    if any(mask):
        plt.scatter(new_images_tsne[mask, 0], new_images_tsne[mask, 1], color=colors(i), label='Cluster %d' % i)
plt.legend()
ax = plt.gca()
ax.axis('off')
plt.suptitle("DBSCAN聚类结果可视化")

# 若无则创建result文件夹
import os
if not os.path.exists('./result'):
    os.mkdir('./result')

# 保存聚类结果图片
plt.savefig('./result/mnist_dbscan_cluster.png')

plt.show()

