# 12.2 数据准备
# 12.2.2 载入分析所需要的库和模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# 12.2.3 变量设置及数据处理
data = pd.read_csv('数据12.1.csv')
X = data.iloc[:, [1,3,4,8]]  # 设置分析所需要的特征变量

# 输出数据集的信息
print("数据集信息：")
print(X.info())

# 输出特征变量的数量
print("特征变量的数量:", len(X.columns))

# 输出特征变量的名称
print("特征变量名称:", X.columns)

# 输出数据集的形状
print("数据集形状:", X.shape)

# 输出数据类型
print("数据类型:", X.dtypes)

# 检查数据集中是否有缺失值并输出缺失值的总数
print("是否有缺失值:", X.isnull().values.any())
print("缺失值总数:", X.isnull().sum())

# 输出数据集的前10行
print("数据集前10行：")
print(X.head(10))

scaler = StandardScaler()
scaler.fit(X)
X_s = scaler.transform(X)
X_s = pd.DataFrame(X_s, columns=X.columns)
# 设置 Pandas 显示选项，显示所有列
pd.set_option('display.max_columns', None)
# 设置 NumPy 打印选项，禁用科学计数法
np.set_printoptions(suppress=True, precision=2)


# 输出相关性矩阵
print("相关性矩阵：")
print(X_s.corr(method='pearson'))

# 绘制相关性热力图
plt.subplot(1, 1, 1)
sns.heatmap(X_s.corr(), annot=True)
plt.show()

# 12.3 划分聚类分析算法
# 12.3.1 使用K均值聚类分析方法对样本示例进行聚类 (K=2)
model = KMeans(n_clusters=2, random_state=2)
model.fit(X_s)
print("K=2的K均值聚类标签：", model.labels_)
print("K=2的K均值聚类结果：")
print(pd.DataFrame(model.labels_.T, index=data.V1, columns=['聚类']).T)
print("K=2的K均值聚类中心：", model.cluster_centers_)
print("K=2的K均值惯性（聚类内误差平方和）：", model.inertia_)

# 12.3.2 使用K均值聚类分析方法对样本示例进行聚类 (K=3)
model = KMeans(n_clusters=3, random_state=2)
model.fit(X_s)
print("K=3的K均值聚类标签：", model.labels_)
print("K=3的K均值聚类结果：")
print(pd.DataFrame(model.labels_.T, index=data.V1, columns=['聚类']).T)
print("K=3的K均值聚类中心：", model.cluster_centers_)
print("K=3的K均值惯性（聚类内误差平方和）：", model.inertia_)

# 12.3.3 使用K均值聚类分析方法对样本示例进行聚类 (K=4)
model = KMeans(n_clusters=4, random_state=3)
model.fit(X_s)
print("K=4的K均值聚类标签：", model.labels_)
print("K=4的K均值聚类结果：")
print(pd.DataFrame(model.labels_.T, index=data.V1, columns=['聚类']).T)
print("K=4的K均值聚类中心：", model.cluster_centers_)
print("K=4的K均值惯性（聚类内误差平方和）：", model.inertia_)

# 12.4 层次聚类方法
# 12.4.1 最短联结法聚类分析
linkage_matrix = linkage(X_s, 'single')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决图表中中文显示问题
dendrogram(linkage_matrix)
plt.title('最短联结法聚类分析树状图')
plt.show()
print("最短联结法聚类联结矩阵：")
print(np.round(linkage_matrix,2))

model = AgglomerativeClustering(n_clusters=3, linkage='single')
model.fit(X_s)
print("最短联结法聚类标签：", model.labels_)
print("最短联结法聚类结果：")
print(pd.DataFrame(model.labels_.T, index=data.V1, columns=['聚类']).T)

# 12.4.2 最长联结法聚类分析
linkage_matrix = linkage(X_s, 'complete')
dendrogram(linkage_matrix)
plt.title('最长联结法聚类分析树状图')
plt.show()
print("最长联结法聚类联结矩阵：")
print(np.round(linkage_matrix,2))

model = AgglomerativeClustering(n_clusters=3, linkage='complete')
model.fit(X_s)
print("最长联结法聚类标签：", model.labels_)
print("最长联结法聚类结果：")
print(pd.DataFrame(model.labels_.T, index=data.V1, columns=['聚类']).T)

# 12.4.3 平均联结法聚类分析
linkage_matrix = linkage(X_s, 'average')
dendrogram(linkage_matrix)
plt.title('平均联结法聚类分析树状图')
plt.show()
print("平均联结法聚类联结矩阵：")
print(np.round(linkage_matrix,2))

model = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='average')
model.fit(X_s)
print("平均联结法聚类标签（欧几里得距离）：", model.labels_)
print("平均联结法聚类结果（欧几里得距离）：")
print(pd.DataFrame(model.labels_.T, index=data.V1, columns=['聚类']).T)

model = AgglomerativeClustering(n_clusters=3, metric='manhattan', linkage='average')
model.fit(X_s)
print("平均联结法聚类标签（曼哈顿距离）：", model.labels_)
print("平均联结法聚类结果（曼哈顿距离）：")
print(pd.DataFrame(model.labels_.T, index=data.V1, columns=['聚类']).T)

model = AgglomerativeClustering(n_clusters=3, metric='cosine', linkage='average')
model.fit(X_s)
print("平均联结法聚类标签（余弦距离）：", model.labels_)
print("平均联结法聚类结果（余弦距离）：")
print(pd.DataFrame(model.labels_.T, index=data.V1, columns=['聚类']).T)

# 12.4.4 ward联结法聚类分析
linkage_matrix = linkage(X_s, 'ward')
dendrogram(linkage_matrix)
plt.title('ward联结法聚类分析树状图')
plt.show()
print("ward联结法聚类联结矩阵：")
print(np.round(linkage_matrix,2))

model = AgglomerativeClustering(n_clusters=3, linkage='ward')
model.fit(X_s)
print("ward联结法聚类标签：", model.labels_)
print("ward联结法聚类结果：")
print(pd.DataFrame(model.labels_.T, index=data.V1, columns=['聚类']).T)

# 12.4.5 重心联结法聚类分析
linkage_matrix = linkage(X_s, 'centroid')
dendrogram(linkage_matrix)
plt.title('重心联结法聚类分析树状图')
plt.show()
print("重心联结法聚类联结矩阵：")
print(np.round(linkage_matrix,2))

labels = fcluster(linkage_matrix, t=3, criterion='maxclust')
print("重心联结法聚类标签：", labels)
print("重心联结法聚类结果：")
print(pd.DataFrame(labels.T, index=data.V1, columns=['聚类']).T)