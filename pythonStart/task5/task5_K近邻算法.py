# K近邻算法
# 2.2 载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import mean_squared_error
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import RocCurveDisplay

# 1. 以交易金额（V1）为响应变量，构建 K 近邻回归算法模型
# 1.1 载入分析所需要的库和模块
# 代码中已经在开头部分完成了库和模块的导入

# 1.2 变量设置及数据处理
data_regression = pd.read_csv('数据9.2.csv')
# 设置特征变量，即除 V1 之外的全部变量
X_regression = data_regression.drop(['V1'], axis=1)
# 设置响应变量，即 V1
y_regression = data_regression['V1']
# 将数据分割为训练集和测试集，test_size=0.3 表示测试集占比 30%，random_state=123 确保结果可复现
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_regression, y_regression, test_size=0.3, random_state=123)
# 创建标准化器对象
scaler_regression = StandardScaler()
# 在训练集上拟合标准化器
scaler_regression.fit(X_train_regression)
# 使用标准化器对训练集和测试集进行标准化
X_train_s_regression = scaler_regression.transform(X_train_regression)
X_test_s_regression = scaler_regression.transform(X_test_regression)

# 1.3 以 K 能取到的最小值、最大值、中间值分别构建 K 近邻回归算法模型
# K 近邻算法(K=1)
model_min = KNeighborsRegressor(n_neighbors=1)
# 在训练集上拟合模型
model_min.fit(X_train_s_regression, y_train_regression)
# 对测试集进行预测
pred_min = model_min.predict(X_test_s_regression)
# 计算均方误差
print("K=1 时的均方误差：", mean_squared_error(y_test_regression, pred_min))
# 计算模型得分
print("K=1 时的模型得分：", model_min.score(X_test_s_regression, y_test_regression))

# K 近邻算法(K=最大值，假设这里数据量决定最大值为 X_train_regression.shape[0]))
max_k = X_train_regression.shape[0]
model_max = KNeighborsRegressor(n_neighbors=max_k)
model_max.fit(X_train_s_regression, y_train_regression)
pred_max = model_max.predict(X_test_s_regression)
print(f"K={max_k} 时的均方误差：", mean_squared_error(y_test_regression, pred_max))
print(f"K={max_k} 时的模型得分：", model_max.score(X_test_s_regression, y_test_regression))

# K 近邻算法(K=中间值，假设这里数据量决定中间值为 X_train_regression.shape[0] // 2))
mid_k = X_train_regression.shape[0] // 2
model_mid = KNeighborsRegressor(n_neighbors=mid_k)
model_mid.fit(X_train_s_regression, y_train_regression)
pred_mid = model_mid.predict(X_test_s_regression)
print(f"K={mid_k} 时的均方误差：", mean_squared_error(y_test_regression, pred_mid))
print(f"K={mid_k} 时的模型得分：", model_mid.score(X_test_s_regression, y_test_regression))

# 1.4 选择最优的 K 值，利用最优 K 值构建 K 近邻回归算法模型
scores = []
ks = range(1, max_k)
for k in ks:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train_s_regression, y_train_regression)
    # 计算模型在测试集上的得分
    score = model.score(X_test_s_regression, y_test_regression)
    scores.append(score)
# 找到最大的得分
max_score = max(scores)
# 找到最大得分对应的索引
index_max = np.argmax(scores)
optimal_k = ks[index_max]
print(f'最优 K 值: {optimal_k}')

# K 近邻算法(选取最优 K 的图形展示)
plt.rcParams['font.sans-serif'] = ['SimHei']
# 绘制 K 值与得分的关系图
plt.plot(ks, scores, 'o-')
plt.xlabel('K')
# 在最优 K 值处绘制垂直线
plt.axvline(optimal_k, linewidth=1, linestyle='--', color='k')
plt.ylabel('拟合优度')
plt.title('不同 K 取值下的拟合优度')
plt.tight_layout()
plt.show()

# 1.5 图形化展示最优模型拟合效果
model_optimal = KNeighborsRegressor(n_neighbors=optimal_k)
model_optimal.fit(X_train_s_regression, y_train_regression)
pred_optimal = model_optimal.predict(X_test_s_regression)
print("最优 K 值时的均方误差：", mean_squared_error(y_test_regression, pred_optimal))
print("最优 K 值时的模型得分：", model_optimal.score(X_test_s_regression, y_test_regression))
t = np.arange(len(y_test_regression))
plt.rcParams['font.sans-serif'] = ['SimHei']
# 绘制原值曲线
plt.plot(t, y_test_regression, 'r-', linewidth=2, label=u'原值')
# 绘制预测值曲线
plt.plot(t, pred_optimal, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# 2. 以“V1 征信违约记录”为响应变量，构建 K 近邻分类算法模型
# 2.1 载入分析所需要的库和模块
# 代码中已经在开头部分完成了库和模块的导入

# 2.2 变量设置及数据处理
data_classification = pd.read_csv('数据5.1.csv')
# 选择特定的特征变量
X_classification = data_classification[['V2', 'V6', 'V7', 'V8', 'V9']]
y_classification = data_classification['V1']
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_classification, test_size=0.3, random_state=123)
scaler_classification = StandardScaler()
scaler_classification.fit(X_train_classification)
X_train_s_classification = scaler_classification.transform(X_train_classification)
X_test_s_classification = scaler_classification.transform(X_test_classification)

# 2.3 以 K 能取到的最小值、最大值、中间值分别构建 K 近邻分类算法模型
# K 近邻算法(K=1)
model_min_class = KNeighborsClassifier(n_neighbors=1)
model_min_class.fit(X_train_s_classification, y_train_classification)
print("K=1 时的模型得分：", model_min_class.score(X_test_s_classification, y_test_classification))

# K 近邻算法(K=最大值，假设这里数据量决定最大值为 X_train_classification.shape[0]))
max_k_class = X_train_classification.shape[0]
model_max_class = KNeighborsClassifier(n_neighbors=max_k_class)
model_max_class.fit(X_train_s_classification, y_train_classification)
print(f"K={max_k_class} 时的模型得分：", model_max_class.score(X_test_s_classification, y_test_classification))

# K 近邻算法(K=中间值，假设这里数据量决定中间值为 X_train_classification.shape[0] // 2))
mid_k_class = X_train_classification.shape[0] // 2
model_mid_class = KNeighborsClassifier(n_neighbors=mid_k_class)
model_mid_class.fit(X_train_s_classification, y_train_classification)
print(f"K={mid_k_class} 时的模型得分：", model_mid_class.score(X_test_s_classification, y_test_classification))

# 2.4 选择最优的 K 值，利用最优 K 值构建 K 近邻分类算法模型
scores_class = []
ks_class = range(1, max_k_class)
for k in ks_class:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_s_classification, y_train_classification)
    score = model.score(X_test_s_classification, y_test_classification)
    scores_class.append(score)
max_score_class = max(scores_class)
index_max_class = np.argmax(scores_class)
optimal_k_class = ks_class[index_max_class]
print(f'最优 K 值: {optimal_k_class}')

# K 近邻算法(选取最优 K 的图形展示)
plt.clf()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(ks_class, scores_class, 'o-')
plt.xlabel('K')
plt.axvline(optimal_k_class, linewidth=1, linestyle='--', color='k')
plt.ylabel('预测准确率')
plt.title('不同 K 取值下的预测准确率')
plt.tight_layout()
plt.show()

# 2.5 图形化展示最优模型拟合效果
model_optimal_class = KNeighborsClassifier(n_neighbors=optimal_k_class)
model_optimal_class.fit(X_train_s_classification, y_train_classification)
pred_optimal_class = model_optimal_class.predict(X_test_s_classification)
t = np.arange(len(y_test_classification))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(t, y_test_classification, 'r-', linewidth=2, label=u'原值')
plt.plot(t, pred_optimal_class, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# 2.6 绘制 K 近邻分类算法 ROC 曲线
scaler_classification_all = StandardScaler()
scaler_classification_all.fit(X_classification)
X_s_classification_all = scaler_classification_all.transform(X_classification)
plt.rcParams['font.sans-serif'] = ['SimHei']
RocCurveDisplay.from_estimator(model_optimal_class, X_s_classification_all, y_classification)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)
plt.title('K 近邻算法 ROC 曲线')
plt.show()

# 2.7 运用“V2 资产负债率”“V8 银行负债”两个特征变量绘制 K 近邻算法决策边界图
X2_classification = X_classification[['V2', 'V8']]
model_decision_boundary = KNeighborsClassifier(n_neighbors=optimal_k_class)
scaler_decision_boundary = StandardScaler()
scaler_decision_boundary.fit(X2_classification)
X2_s_classification = scaler_decision_boundary.transform(X2_classification)
model_decision_boundary.fit(X2_s_classification, y_classification)
print("使用两个特征变量的模型得分：", model_decision_boundary.score(X2_s_classification, y_classification))
plt.rcParams['font.sans-serif'] = ['SimHei']
plot_decision_regions(np.array(X2_s_classification), np.array(y_classification), model_decision_boundary)
plt.xlabel('资产负债率')
plt.ylabel('银行负债')
plt.title('K 近邻算法决策边界')
plt.show()

# 2.8 普通 KNN 算法、带权重 KNN、指定半径 KNN 三种算法分别构建模型并进行对比
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=optimal_k_class)))
models.append(('KNN with weights', KNeighborsClassifier(n_neighbors=optimal_k_class, weights='distance')))
models.append(('Radius Neighbors', RadiusNeighborsClassifier(radius=100)))
# 基于验证集法
results = []
for name, model in models:
    model.fit(X_train_s_classification, y_train_classification)
    results.append((name, model.score(X_test_s_classification, y_test_classification)))
for i in range(len(results)):
    print('name: {}; score: {}'.format(results[i][0], results[i][1]))

# 基于 10 折交叉验证法
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=optimal_k_class)))
models.append(('KNN with weights', KNeighborsClassifier(n_neighbors=optimal_k_class, weights='distance')))
models.append(('Radius Neighbors', RadiusNeighborsClassifier(radius=10000)))
results = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, X_s_classification_all, y_classification, cv=kfold)
    results.append((name, cv_result.mean()))
for i in range(len(results)):
    print('name: {}; cross_val_score: {}'.format(results[i][0], results[i][1]))