# 多元Logistic回归算法

# 2.1 载入分析所需要的库和模块
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

# 2.2 数据读取及观察
data = pd.read_csv('数据6.1.csv')
# 输出数据的信息，包括列数据类型、非空值数量等
print("数据信息:")
print(data.info())
# 输出数据的列数
print("数据列数:")
print(len(data.columns))
# 输出数据的列名
print("数据列名:")
print(data.columns)
# 输出数据的形状（行数和列数）
print("数据形状:")
print(data.shape)
# 输出数据各列的数据类型
print("数据各列数据类型:")
print(data.dtypes)
# 检查数据中是否存在缺失值，返回布尔值
print("数据是否存在缺失值:")
print(data.isnull().values.any())
# 输出数据每列的缺失值数量
print("数据每列缺失值数量:")
print(data.isnull().sum())
# 输出数据的前几行（默认5行）
print("数据前几行:")
print(data.head())
# 输出变量V1的取值计数
print("V1的取值计数:")
print(data.V1.value_counts())

# 3 描述性分析及图形绘制
# 3.1 描述性分析
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# 计算连续变量的统计指标（均值、标准差、最小值、最大值、四分位数等）并输出
print("连续变量统计指标:")
print(data.describe())

# 3.2 绘制条形图
# 绘制职称情况（V5）变量的条形图，用于展示不同职称的数量分布
print("绘制职称情况（V5）条形图:")
sns.countplot(x=data['V5'])
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title("职称情况条形图")
plt.show()

# 3.3 绘制箱线图
# 绘制职称情况（V5）与工作年限（V2）的箱线图
print("绘制职称情况（V5）与工作年限（V2）箱线图:")
sns.boxplot(x='V5', y='V2', hue='V5', data=data, palette="Blues", legend=False)
plt.show()

# 绘制职称情况（V5）与绩效考核得分（V3）的箱线图
print("绘制职称情况（V5）与绩效考核得分（V3）箱线图:")
sns.boxplot(x='V5', y='V3', hue='V5', data=data, palette="Blues", legend=False)
plt.show()

# 绘制职称情况（V5）与违规操作积分（V4）的箱线图
print("绘制职称情况（V5）与违规操作积分（V4）箱线图:")
sns.boxplot(x='V5', y='V4', hue='V5', data=data, palette="Blues", legend=False)
plt.show()

# 4 数据处理
# 4.1 区分分类特征和连续特征并进行处理
def data_encoding(data):
    data = data[["V5", 'V2', "V3", "V4"]]
    Discretefeature = []
    Continuousfeature = ['V2', "V3", "V4"]
    df = pd.get_dummies(data, columns=Discretefeature)
    df[Continuousfeature] = (df[Continuousfeature] - df[Continuousfeature].mean()) / (df[Continuousfeature].std())
    df["V5"] = data[["V5"]]
    return df

data = data_encoding(data)

# 4.2 将样本示例全集分割为训练样本和测试样本
X = data.drop(['V5'], axis=1)  # 设置特征变量，即除V5之外的全部变量
y = data['V5']  # 设置响应变量，即V5
# 将数据分割为训练集和测试集，并输出训练集特征变量、测试集特征变量、训练集响应变量、测试集响应变量的形状
print("分割训练集和测试集:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)
print("训练集特征变量形状:", X_train.shape)
print("测试集特征变量形状:", X_test.shape)
print("训练集响应变量形状:", y_train.shape)
print("测试集响应变量形状:", y_test.shape)

# 5 建立多元Logistic回归算法模型
# 5.1 模型估计
model = LogisticRegression(multi_class='multinomial', solver='newton-cg', C=1e10, max_iter=1000)
model.fit(X_train, y_train)
# 输出模型的迭代次数
print("模型迭代次数:")
print(model.n_iter_)
# 输出模型的截距项
print("模型截距项:")
print(model.intercept_)
# 输出模型的回归系数
print("模型回归系数:")
print(model.coef_)

# 5.2 模型性能分析
# 计算模型预测的准确率并输出
print("模型预测准确率:")
print(model.score(X_test, y_test))
# 计算响应变量预测分类概率并输出前5个样本示例的概率
print("前5个样本示例的预测分类概率:")
prob = model.predict_proba(X_test)
print(prob[:5])

np.set_printoptions(suppress=True)
prob = model.predict_proba(X_test)
print("前5个样本示例的预测分类概率（抑制科学计数法）:")
print(prob[:5])

pred = model.predict(X_test)
print("前5个样本示例的预测分类类型:")
print(pred[:5])

# 基于测试样本输出混淆矩阵并输出
print("混淆矩阵:")
table = confusion_matrix(y_test, pred)
print(table)
# 输出混淆矩阵热图
print("混淆矩阵热图:")
sns.heatmap(table, cmap='Reds', annot=True)
plt.tight_layout()
plt.show()
# 输出详细的预测效果指标
print("详细预测效果指标:")
print(classification_report(y_test, pred))
# 计算kappa得分并输出
print("Kappa得分:")
print(cohen_kappa_score(y_test, pred))