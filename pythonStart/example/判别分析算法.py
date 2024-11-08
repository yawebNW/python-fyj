from mlxtend.plotting import plot_decision_regions

# 判别分析算法
# 2.1 载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

# 2.2 线性判别分析降维优势展示
# 绘制三维数据的分布图
X, y = make_classification(n_samples=500, n_features=3, n_redundant=0,
                           n_classes=3, n_informative=2, n_clusters_per_class=1,
                           class_sep=0.5, random_state=100)  # 生成三类三维特征的数据
plt.rcParams['axes.unicode_minus'] = False  # 解决负号不显示问题
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', c=y)
# 使用 PCA 进行降维
pca = PCA(n_components=2)
pca.fit(X)
# 打印解释方差比例和解释方差
print("PCA 解释方差比例：", pca.explained_variance_ratio_)
print("PCA 解释方差：", pca.explained_variance_)
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
plt.show()
# 使用 LDA 进行降维
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
X_new = lda.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
plt.show()  # 降维后样本特征信息之间的关系得以保留

# 2.3 数据读取及观察
data = pd.read_csv('数据7.1.csv')
# 打印数据信息
print("数据信息：", data.info())
print("列数：", len(data.columns))
print("列名：", data.columns)
print("形状：", data.shape)
print("数据类型：", data.dtypes)
print("是否有空值？", data.isnull().values.any())
print("空值数量：", data.isnull().sum())
print("前几行数据：", data.head())
print("V1 的值计数：", data.V1.value_counts())

# 3 特征变量相关性分析
X = data.drop(['V1'], axis=1)  # 设置特征变量，即除 V1 之外的全部变量
y = data['V1']  # 设置响应变量，即 V1
# 打印相关性矩阵
print("相关性矩阵：", X.corr())
sns.heatmap(X.corr(), cmap='Blues', annot=True)

# 4 使用样本示例全集开展线性判别分析
# 4.1 模型估计及性能分析
# 使用样本示例全集开展 LDA
model = LinearDiscriminantAnalysis()  # 使用 LDA 算法
model.fit(X, y)  # 使用 fit 方法进行拟合
# 打印模型在全数据上的得分
print("模型在全数据上的得分：", model.score(X, y))
print("先验概率：", model.priors_)
print("各类均值：", model.means_)
np.set_printoptions(suppress=True)  # 不以科学计数法显示，而是直接显示数字
print("系数：", model.coef_)
print("截距项：", model.intercept_)
print("可解释方差比例：", model.explained_variance_ratio_)
print("缩放系数：", model.scalings_)

lda_scores = model.fit(X, y).transform(X)
# 打印 LDA 得分的形状和前几行
print("LDA 得分形状：", lda_scores.shape)
print("前几行 LDA 得分：", lda_scores[:5, :])

LDA_scores = pd.DataFrame(lda_scores, columns=['LD1', 'LD2'])
LDA_scores['网点类型'] = data['V1']
# 打印带有标签的 LDA 得分的前几行
print("带有标签的前几行 LDA 得分：", LDA_scores.head())

d = {0: '未转型网点', 1: '一般网点', 2: '精品网点'}
LDA_scores['网点类型'] = LDA_scores['网点类型'].map(d)
# 打印映射后的前几行
print("映射后的前几行：", LDA_scores.head())

plt.rcParams['axes.unicode_minus'] = False  # 解决图表中负号不显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决图表中中文显示问题。
sns.scatterplot(x='LD1', y='LD2', data=LDA_scores, hue='网点类型')

# 4.2 运用两个特征变量绘制 LDA 决策边界图
X2 = X.iloc[:, 0:2]  # 仅选取 V2 存款规模、V3EVA 作为特征变量
model = LinearDiscriminantAnalysis()  # 使用 LDA 算法
model.fit(X2, y)  # 使用 fit 方法进行拟合
# 打印模型在两个特征上的得分和解释方差比例
print("模型在两个特征上的得分：", model.score(X2, y))
print("两个特征上的解释方差比例：", model.explained_variance_ratio_)

plot_decision_regions(np.array(X2), np.array(y), model)
plt.xlabel('存款规模')  # 将 x 轴设置为'存款规模'
plt.ylabel('EVA')  # 将 y 轴设置为'EVA'
plt.title('LDA 决策边界')  # 将标题设置为'LDA 决策边界'
plt.show()

# 5 使用分割样本开展线性判别分析
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)
model = LinearDiscriminantAnalysis()  # 使用 LDA 算法
model.fit(X_train, y_train)  # 基于训练样本使用 fit 方法进行拟合
# 打印模型在测试集上的得分
print("模型在测试集上的得分：", model.score(X_test, y_test))

prob = model.predict_proba(X_test)
# 打印前几个预测概率
print("前几个预测概率：", prob[:5])

pred = model.predict(X_test)
# 打印前几个预测结果
print("前几个预测结果：", pred[:5])

# 打印测试集的混淆矩阵
print("测试集的混淆矩阵：", confusion_matrix(y_test, pred))

# 打印分类报告
print("分类报告：", classification_report(y_test, pred))

# 打印 Cohen kappa 得分
print("Cohen kappa 得分：", cohen_kappa_score(y_test, pred))

# 6 使用分割样本开展二次判别分析
# 6.1 模型估计
model = QuadraticDiscriminantAnalysis()  # 使用 QDA 算法
model.fit(X_train, y_train)  # 基于训练样本使用 fit 方法进行拟合
# 打印模型在测试集上的得分
print("模型在测试集上的得分（QDA）：", model.score(X_test, y_test))

prob = model.predict_proba(X_test)
# 打印前几个预测概率（QDA）
print("前几个预测概率（QDA）：", prob[:5])

pred = model.predict(X_test)
# 打印前几个预测结果（QDA）
print("前几个预测结果（QDA）：", pred[:5])

# 打印测试集的混淆矩阵（QDA）
print("测试集的混淆矩阵（QDA）：", confusion_matrix(y_test, pred))

# 打印分类报告（QDA）
print("分类报告（QDA）：", classification_report(y_test, pred))

# 打印 Cohen kappa 得分（QDA）
print("Cohen kappa 得分（QDA）：", cohen_kappa_score(y_test, pred))

# 6.2 运用两个特征变量绘制 QDA 决策边界图
X2 = X.iloc[:, 0:2]
model = QuadraticDiscriminantAnalysis()
model.fit(X2, y)
# 打印模型在两个特征上的得分（QDA）
print("模型在两个特征上的得分（QDA）：", model.score(X2, y))

plot_decision_regions(np.array(X2), np.array(y), model)
plt.xlabel('存款规模')  # 将 x 轴设置为'存款规模'
plt.ylabel('EVA')  # 将 y 轴设置为'EVA'
plt.title('QDA 决策边界')  # 将标题设置为'QDA 决策边界'
plt.show()