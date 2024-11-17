# 第十七章 神经网络算法

# 17.2.2 载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import RocCurveDisplay

# 17.5 多分类神经网络算法示例
# 17.5.1 变量设置及数据处理
data = pd.read_csv('数据6.1.csv')
X = data.iloc[:, 1:4]  # 设置特征变量
y = data.iloc[:, 0]  # 设置响应变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=10)

# 数据标准化
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
X_train_s = pd.DataFrame(X_train_s, columns=X_train.columns)
X_test_s = pd.DataFrame(X_test, columns=X_test.columns)

# 17.5.2 单隐藏层多分类问题神经网络算法
model = MLPClassifier(solver='sgd', learning_rate_init=0.01, learning_rate='constant', tol=0.0001, activation='relu', hidden_layer_sizes=(3,), random_state=10, max_iter=2000)
model.fit(X_train_s, y_train)
score = model.score(X_test_s, y_test)
print(f"单隐藏层多分类问题神经网络算法测试集得分: {score:.3f}")

# 17.5.3 双隐藏层多分类问题神经网络算法
model = MLPClassifier(solver='sgd', learning_rate_init=0.01, learning_rate='constant', tol=0.0001, activation='relu', hidden_layer_sizes=(3, 2), random_state=10, max_iter=2000)
model.fit(X_train_s, y_train)
score = model.score(X_test_s, y_test)
print(f"双隐藏层多分类问题神经网络测试集得分: {score:.3f}")

# 17.5.4 模型性能评价
pred = model.predict(X_test_s)
print(f"预测结果前5个样本:\n{pred[:5]}")
print(f"混淆矩阵:\n{confusion_matrix(y_test, pred)}")
print(classification_report(y_test, pred))
print(f"Kappa得分: {cohen_kappa_score(y_test, pred):.3f}")

# 17.5.5 运用两个特征变量绘制多分类神经网络算法决策边界图
X2_train_s = X_train_s.iloc[:, [0,1]]  # 仅选取特定特征变量
X2_test_s = X_test_s.iloc[:, [0,1]]  # 仅选取特定特征变量
model = MLPClassifier(solver='sgd', learning_rate_init=0.01, learning_rate='constant', tol=0.0001, activation='relu', hidden_layer_sizes=(3, 2), random_state=10, max_iter=2000)
model.fit(X2_train_s, y_train)
score = model.score(X2_test_s, y_test)
print(f"多分类神经网络算法测试集得分: {score:.3f}")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决图表中中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决图表中负号不显示问题
plot_decision_regions(np.array(X2_test_s), np.array(y_test), model)
plt.xlabel('工作年限')
plt.ylabel('绩效考核得分')
plt.title('多分类神经网络算法决策边界')  # 将标题设置为'多分类神经网络算法决策边界'
plt.show()