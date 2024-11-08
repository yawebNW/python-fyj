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

# 2.3 数据读取及观察（针对数据6.1文件）
data_6_1 = pd.read_csv('数据6.1.csv')

print("数据6.1信息：")
print(data_6_1.info())
print("\n列数：")
print(len(data_6_1.columns))
print("\n列名：")
print(data_6_1.columns)
print("\n形状：")
print(data_6_1.shape)
print("\n数据类型：")
print(data_6_1.dtypes)
print("\n是否有空值？")
print(data_6_1.isnull().values.any())
print("\n空值数量：")
print(data_6_1.isnull().sum())
print("\n前几行数据：")
print(data_6_1.head())

# 3 特征变量相关性分析（针对数据6.1文件）
X_6_1 = data_6_1.drop(['V1'], axis=1)  # 设置特征变量，即除 V1 之外的全部变量
y_6_1 = data_6_1['V1']  # 设置响应变量，即 V1

print("数据6.1相关性矩阵：")
print(X_6_1.corr())
sns.heatmap(X_6_1.corr(), cmap='Blues', annot=True)

# 4 使用样本示例全集开展线性判别分析（针对数据6.1文件）
# 4.1 模型估计及性能分析
# 使用样本示例全集开展 LDA
model_6_1 = LinearDiscriminantAnalysis()  # 使用 LDA 算法
model_6_1.fit(X_6_1, y_6_1)  # 使用 fit 方法进行拟合

print("数据6.1模型在全数据上的得分：")
print(model_6_1.score(X_6_1, y_6_1))
print("\n先验概率：")
print(model_6_1.priors_)
print("\n各类均值：")
print(model_6_1.means_)
np.set_printoptions(suppress=True)  # 不以科学计数法显示，而是直接显示数字
print("\n系数：")
print(model_6_1.coef_)
print("\n截距项：")
print(model_6_1.intercept_)
print("\n可解释方差比例：")
print(model_6_1.explained_variance_ratio_)
print("\n缩放系数：")
print(model_6_1.scalings_)

lda_scores_6_1 = model_6_1.fit(X_6_1, y_6_1).transform(X_6_1)

print("数据6.1 LDA 得分形状：")
print(lda_scores_6_1.shape)
print("\n数据6.1前几行 LDA 得分：")
print(lda_scores_6_1[:5, :])

LDA_scores_6_1 = pd.DataFrame(lda_scores_6_1, columns=['LD1', 'LD2'])
LDA_scores_6_1['网点类型'] = data_6_1['V1']

print("数据6.1带有标签的前几行 LDA 得分：")
print(LDA_scores_6_1.head())

d = {0: '未转型网点', 1: '一般网点', 2: '精品网点'}
LDA_scores_6_1['网点类型'] = LDA_scores_6_1['网点类型'].map(d)

print("数据6.1映射后的前几行：")
print(LDA_scores_6_1.head())

plt.rcParams['axes.unicode_minus'] = False  # 解决图表中负号不显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决图表中中文显示问题。
sns.scatterplot(x='LD1', y='LD2', data=LDA_scores_6_1, hue='网点类型')
plt.show()

# 4.2 运用两个特征变量绘制 LDA 决策边界图（针对数据6.1文件）
X2_6_1 = X_6_1.iloc[:, 0:2]  # 仅选取前两个特征变量
model_6_1 = LinearDiscriminantAnalysis()  # 使用 LDA 算法
model_6_1.fit(X2_6_1, y_6_1)  # 使用 fit 方法进行拟合

print("数据6.1模型在两个特征上的得分：")
print(model_6_1.score(X2_6_1, y_6_1))
print("\n数据6.1两个特征上的解释方差比例：")
print(model_6_1.explained_variance_ratio_)

plot_decision_regions(np.array(X2_6_1), np.array(y_6_1), model_6_1)
plt.xlabel('收入档次')  # 将 x 轴设置为特征名称
plt.ylabel('工作年限')  # 将 y 轴设置为特征名称
plt.title('数据6.1 LDA 决策边界')  # 将标题设置为'LDA 决策边界'
plt.show()

# 5 使用分割样本开展线性判别分析（针对数据6.1文件）
X_train_6_1, X_test_6_1, y_train_6_1, y_test_6_1 = train_test_split(X_6_1, y_6_1, test_size=0.3, stratify=y_6_1, random_state=123)
model_6_1 = LinearDiscriminantAnalysis()  # 使用 LDA 算法
model_6_1.fit(X_train_6_1, y_train_6_1)  # 基于训练样本使用 fit 方法进行拟合

print("数据6.1模型在测试集上的得分：")
print(model_6_1.score(X_test_6_1, y_test_6_1))

prob_6_1 = model_6_1.predict_proba(X_test_6_1)

print("数据6.1前几个预测概率：")
print(prob_6_1[:5])

pred_6_1 = model_6_1.predict(X_test_6_1)

print("数据6.1前几个预测结果：")
print(pred_6_1[:5])

print("数据6.1测试集的混淆矩阵：")
print(confusion_matrix(y_test_6_1, pred_6_1))

print("数据6.1分类报告：")
print(classification_report(y_test_6_1, pred_6_1))

print("数据6.1 Cohen kappa 得分：")
print(cohen_kappa_score(y_test_6_1, pred_6_1))

# 针对数据5.1文件
data_5_1 = pd.read_csv('数据5.1.csv')
# 把响应变量设为“V1征信违约记录”，将其他变量作为特征变量
X_5_1 = data_5_1.drop(['V1'], axis=1)
y_5_1 = data_5_1['V1']

# 使用分割样本开展二次判别分析
X_train_5_1, X_test_5_1, y_train_5_1, y_test_5_1 = train_test_split(X_5_1, y_5_1, test_size=0.3, stratify=y_5_1, random_state=123)

# 6.1 模型估计
model_5_1 = QuadraticDiscriminantAnalysis()  # 使用 QDA 算法
model_5_1.fit(X_train_5_1, y_train_5_1)  # 基于训练样本使用 fit 方法进行拟合

print("数据5.1模型在测试集上的得分（QDA）：")
print(model_5_1.score(X_test_5_1, y_test_5_1))

prob_5_1 = model_5_1.predict_proba(X_test_5_1)

print("数据5.1前几个预测概率（QDA）：")
print(prob_5_1[:5])

pred_5_1 = model_5_1.predict(X_test_5_1)

print("数据5.1前几个预测结果（QDA）：")
print(pred_5_1[:5])

print("数据5.1测试集的混淆矩阵（QDA）：")
print(confusion_matrix(y_test_5_1, pred_5_1))

print("数据5.1分类报告（QDA）：")
print(classification_report(y_test_5_1, pred_5_1))

print("数据5.1 Cohen kappa 得分（QDA）：")
print(cohen_kappa_score(y_test_5_1, pred_5_1))

# 6.2 运用两个特征变量绘制 QDA 决策边界图（针对数据5.1文件）
X2_5_1 = X_5_1.iloc[:, 0:2]
model_5_1 = QuadraticDiscriminantAnalysis()
model_5_1.fit(X2_5_1, y_5_1)

print("数据5.1模型在两个特征上的得分（QDA）：")
print(model_5_1.score(X2_5_1, y_5_1))

plot_decision_regions(np.array(X2_5_1), np.array(y_5_1), model_5_1)
plt.xlabel('V1征信违约记录')  # 将 x 轴设置为特征名称
plt.ylabel('V2')  # 将 y 轴设置为特征名称
plt.title('数据5.1 QDA 决策边界')  # 将标题设置为'QDA 决策边界'
plt.show()