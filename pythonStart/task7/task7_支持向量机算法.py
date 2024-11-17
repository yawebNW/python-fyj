#  支持向量机算法
#16.2.2  载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions

# 第十六章 多分类支持向量机算法示例

# 16.5.1 变量设置及数据处理
data = pd.read_csv('数据6.1.csv')
X = data.iloc[:, 1:4]  # 设置特征变量
y = data.iloc[:, 0]  # 设置响应变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=10)

# 数据标准化
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# 16.5.2 多分类支持向量机算法（一对一）
classifier = svm.SVC(C=1000, kernel='rbf', gamma='scale', decision_function_shape='ovo')  # 使用一对多策略
classifier.fit(X_train_s, y_train)
print(f"训练集：{classifier.score(X_train_s, y_train):.3f}")  # 计算分类器对训练集的准确率
print(f"测试集：{classifier.score(X_test_s, y_test):.3f}")  # 计算分类器对测试集的准确率
print(f"内部决策函数:\n{classifier.decision_function(X_train_s)}")  #查看内部决策函数，返回的是样本到超平面的距离
print(f"预测结果：\n{classifier.predict(X_train_s)}")  # 查看预测结果

# 运用两个特征变量绘制多分类支持向量机算法决策边界图
X2 = X.iloc[:, [0, 1]]  # 仅选取工作年限、绩效考核得分作为特征变量
X2_s =scaler.fit_transform(X2)
model = SVC(kernel="rbf", C=10, random_state=10)
model.fit(X2_s, y)
model.score(X2_s, y)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决图表中中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决图表中负号不显示问题
plot_decision_regions(np.array(X2_s), np.array(y), model)
plt.xlabel('工作年限')  # 将x轴设置为'工作年限'
plt.ylabel('绩效考核得分')  # 将y轴设置为'绩效考核得分'
plt.title('多分类支持向量机算法决策边界')  # 将标题设置为'多分类支持向量机算法决策边界'
plt.show()

# 16.5.3 多分类支持向量机算法（默认参数）
# 线性核函数算法
model = SVC(kernel="linear", random_state=123)
model.fit(X_train_s, y_train)
print(f"线性核函数算法测试集得分: {model.score(X_test_s, y_test):.3f}")

# 多项式核函数算法
model = SVC(kernel="poly", degree=2, random_state=123)
model.fit(X_train_s, y_train)
print(f"多项式核函数算法测试集得分: {model.score(X_test_s, y_test):.3f}")

# 多项式核函数算法
model = SVC(kernel="poly", degree=3, random_state=123)
model.fit(X_train_s, y_train)
print(f"多项式核函数算法测试集得分: {model.score(X_test_s, y_test):.3f}")

# 径向基函数算法
model = SVC(kernel='rbf', random_state=123)
model.fit(X_train_s, y_train)
print(f"径向基函数算法测试集得分: {model.score(X_test_s, y_test):.3f}")

# sigmod核函数算法
model = SVC(kernel="sigmoid", random_state=123)
model.fit(X_train_s, y_train)
print(f"sigmod核函数算法测试集得分: {model.score(X_test_s, y_test):.3f}")

# 16.5.4 通过K折交叉验证寻求最优参数
# 通过K折交叉验证寻求最优参数（线性核函数算法）
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1, 10]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel='linear',random_state=123), param_grid, cv=kfold, n_jobs=-1)
model.fit(X_train_s, y_train)
print(f"线性核函数最优参数: {model.best_params_}，最优模型测试集得分: {model.score(X_test_s, y_test):.3f}")

# 通过K折交叉验证寻求最优参数（多项式核函数）
param_grid = {'C': [0.001, 0.01, 0.1, 10], 'gamma': [0.001, 0.01, 0.1, 1, 10], 'degree': [1, 2, 3]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel='poly', random_state=123), param_grid, cv=kfold, n_jobs=-1)
model.fit(X_train_s, y_train)
print(f"多项式核函数最优参数: {model.best_params_}，最优模型测试集得分: {model.score(X_test_s, y_test):.3f}")

# 通过K折交叉验证寻求最优参数（径向基函数算法）
param_grid = {'C': [0.001, 0.01, 0.1, 10], 'gamma': [0.001, 0.01, 0.1, 1, 10]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel='rbf', random_state=123), param_grid, cv=kfold, n_jobs=-1)
model.fit(X_train_s, y_train)
print(f"径向基函数最优参数: {model.best_params_}，最优模型测试集得分: {model.score(X_test_s, y_test):.3f}")

# 通过K折交叉验证寻求最优参数（sigmod核函数算法）
param_grid = {'C': [0.001, 0.01, 0.1, 10], 'gamma': [0.001, 0.01, 0.1, 1, 10]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel='sigmoid', random_state=123), param_grid, cv=kfold, n_jobs=-1)
model.fit(X_train_s, y_train)
print(f"sigmod核函数最优参数: {model.best_params_}，最优模型测试集得分: {model.score(X_test_s, y_test):.3f}")

# 16.5.5 模型性能评价
pred = model.predict(X_test_s)
print(f"预测结果前5个样本:\n{pred[:5]}")
print(f"混淆矩阵:\n{confusion_matrix(y_test, pred)}")
sns.heatmap(confusion_matrix(y_test, pred), cmap='Blues', annot=True)
plt.tight_layout()
print(f"分类报告:\n{classification_report(y_test, pred)}")
print(f"Kappa得分: {cohen_kappa_score(y_test, pred):.3f}")