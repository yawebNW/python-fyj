# 14.2 随机森林算法
# 14.2.2 载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import RocCurveDisplay
from sklearn.inspection import PartialDependenceDisplay
from mlxtend.plotting import plot_decision_regions

# 14.3 分类问题随机森林算法示例
# 14.3.1 变量设置及数据处理
data = pd.read_csv('数据5.1.csv')
X = data.iloc[:, [1,5,6,7,8]]  # 设置特征变量
y = data.iloc[:, 0]  # 设置响应变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=10)

# 14.3.2 二元Logistic回归、单颗分类决策树算法观察
model = LogisticRegression(C=1e10, max_iter=1000, fit_intercept=True)
model.fit(X_train, y_train)
print("Logistic回归模型在测试集上的得分:", np.round(model.score(X_test, y_test)),3)

# 单颗分类决策树算法
model = DecisionTreeClassifier()
path = model.cost_complexity_pruning_path(X_train, y_train)
param_grid = {'ccp_alpha': path.ccp_alphas}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
model = GridSearchCV(DecisionTreeClassifier(random_state=10), param_grid, cv=kfold)
model.fit(X_train, y_train)
print("最优alpha值：", model.best_params_)
model = model.best_estimator_
print("最优预测准确率：", np.round(model.score(X_test, y_test),3))

# 14.3.3 装袋法分类算法
model = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=10), n_estimators=300, max_samples=0.8, random_state=0)
model.fit(X_train, y_train)
print("装袋法分类算法在测试集上的得分:", np.round(model.score(X_test, y_test),3))

# 14.3.4 随机森林分类算法
model = RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=10)
model.fit(X_train, y_train)
print("随机森林分类算法在测试集上的得分:", np.round(model.score(X_test, y_test),3))

# 14.3.5 寻求max_features最优参数
scores = []
for max_features in range(1, X.shape[1] + 1):
    model = RandomForestClassifier(max_features=max_features, n_estimators=300, random_state=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)
index = np.argmax(scores)
print("最优max_features值：", range(1, X.shape[1] + 1)[index])
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.plot(range(1, X.shape[1] + 1), scores, 'o-')
plt.axvline(range(1, X.shape[1] + 1)[index], linestyle='--', color='k', linewidth=1)
plt.xlabel('最大特征变量数')
plt.ylabel('最优预测准确率')
plt.title('预测准确率随选取的最大特征变量数变化情况')
plt.show()
print("预测准确率随最大特征变量数变化情况：", np.round(scores,3))

# 14.3.6 寻求n_estimators最优参数
ScoreAll = []
for i in range(100, 300, 10):
    model = RandomForestClassifier(max_features=2, n_estimators=i, random_state=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    ScoreAll.append([i, score])
ScoreAll = np.array(ScoreAll)
print("n_estimators与预测准确率的关系：\n", np.round(ScoreAll,3))
max_score = np.where(ScoreAll==np.max(ScoreAll[:, 1]))[0][0]
print("最优参数以及最高得分:", np.round(ScoreAll[max_score],3))
plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
plt.xlabel('n_estimators')
plt.ylabel('预测准确率')
plt.title('预测准确率随n_estimators变化情况')
plt.show()

# 14.3.7 随机森林特征变量重要性水平分析
sorted_index = model.feature_importances_.argsort()
plt.barh(range(X_train.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index])
plt.xlabel('特征变量重要性水平')
plt.ylabel('特征变量')
plt.title('随机森林特征变量重要性水平分析')
plt.tight_layout()
plt.show()

# 14.3.8 绘制部分依赖图与个体条件期望图
PartialDependenceDisplay.from_estimator(model, X_train, ['V2', 'V6'], kind='average')
PartialDependenceDisplay.from_estimator(model, X_train, ['V2', 'V6'], kind='individual')
PartialDependenceDisplay.from_estimator(model, X_train, ['V2', 'V6'], kind='both')

# 14.3.9 模型性能评价
prob = model.predict_proba(X_test)
print("预测概率前5个样本:\n", np.round(prob[:5],3))
pred = model.predict(X_test)
print("预测结果前5个样本:\n", np.round(pred[:5],3))
print("混淆矩阵:\n", confusion_matrix(y_test, pred))
print("分类报告:\n", classification_report(y_test, pred))
print("Kappa得分:", np.round(cohen_kappa_score(y_test, pred),3))

# 14.3.10 绘制ROC曲线
RocCurveDisplay.from_estimator(model, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)
plt.title('随机森林分类树算法ROC曲线')
plt.show()

# 14.3.11 运用两个特征变量绘制随机森林算法决策边界图
X2 = X.iloc[:, [0, 1]]  # 仅选取资产负债率、主营业务收入作为特征变量
model = RandomForestClassifier(n_estimators=300, max_features=1, random_state=1)
model.fit(X2, y)
print("模型预测准确率：", model.score(X2, y))
plot_decision_regions(np.array(X2), np.array(y), model)
plt.xlabel('资产负债率')
plt.ylabel('主营业务收入')
plt.title('随机森林算法决策边界')
plt.show()

# 14.4 回归问题随机森林算法示例
# 14.4.1 变量设置及数据处理
data = pd.read_csv('数据4.3.csv')
X = data.iloc[:, 2:]  # 设置特征变量
y = data.iloc[:, 1]  # 设置响应变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# 14.4.2 线性回归、单颗回归决策树算法观察
model = LinearRegression()
model.fit(X_train, y_train)
print("线性回归模型在测试集上的得分:", np.round(model.score(X_test, y_test),3))

# 单颗回归决策树算法
# model = DecisionTreeRegressor()
# path = model.cost_complexity_pruning_path(X_train, y_train)
# param_grid = {'ccp_alpha': path.ccp_alphas}
# kfold = KFold(n_splits=10, shuffle=True, random_state=10)
# model = GridSearchCV(DecisionTreeRegressor(random_state=10), param_grid, cv=kfold, n_jobs=-1)
# model.fit(X_train, y_train)
# print("最优alpha值：",model.best_params_,3)
# model = model.best_estimator_
# print("最优拟合优度：", np.round(model.score(X_test, y_test),3))

# 14.4.3 装袋法回归算法
model = BaggingRegressor(estimator=DecisionTreeRegressor(random_state=10), n_estimators=300, max_samples=0.9, oob_score=True, random_state=0)
model.fit(X_train, y_train)
print("装袋法回归算法在测试集上的得分:", np.round(model.score(X_test, y_test),3))

# 14.4.4 随机森林回归算法
max_features = int(X_train.shape[1] / 3)
model = RandomForestRegressor(n_estimators=300, max_features=max_features, random_state=10)
model.fit(X_train, y_train)
print("随机森林回归算法在测试集上的得分:", np.round(model.score(X_test, y_test),3))

# 14.4.5 寻求max_features最优参数
scores = []
for max_features in range(1, X.shape[1] + 1):
    model = RandomForestRegressor(max_features=max_features,
                                  n_estimators=300, random_state=123)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)
index = np.argmax(scores)
print("最优max_features值：", np.round(range(1, X.shape[1] + 1)[index],3))
plt.plot(range(1, X.shape[1] + 1), scores, 'o-')
plt.axvline(range(1, X.shape[1] + 1)[index], linestyle='--', color='k', linewidth=1)
plt.xlabel('最大特征变量数')
plt.ylabel('拟合优度')
plt.title('拟合优度随选取的最大特征变量数变化情况')
plt.show()
print("拟合优度随最大特征变量数变化情况：", np.round(scores,3))

# 14.4.6 寻求n_estimators最优参数
ScoreAll = []
for i in range(10, 100, 10):
    model = RandomForestRegressor(max_features=1, n_estimators=i, random_state=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    ScoreAll.append([i, score])
ScoreAll = np.array(ScoreAll)
print("n_estimators与拟合优度的关系：\n",np.round(ScoreAll,3) )
max_score = np.where(ScoreAll==np.max(ScoreAll[:, 1]))[0][0]
print("最优参数以及最高得分:", np.round(ScoreAll[max_score],3))
plt.figure(figsize=[20, 5])
plt.xlabel('n_estimators')
plt.ylabel('拟合优度')
plt.title('拟合优度随n_estimators变化情况')
plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
plt.show()

# 14.4.7 随机森林特征变量重要性水平分析
sorted_index = model.feature_importances_.argsort()
plt.barh(range(X_train.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index])
plt.xlabel('特征变量重要性水平')
plt.ylabel('特征变量')
plt.title('随机森林特征变量重要性水平分析')
plt.tight_layout()
plt.show()

# 14.4.8 绘制部分依赖图与个体条件期望图
PartialDependenceDisplay.from_estimator(model, X_train, ['Net interest income', 'Intermediate income'], kind='average')
PartialDependenceDisplay.from_estimator(model, X_train, ['Net interest income', 'Intermediate income'], kind='individual')
PartialDependenceDisplay.from_estimator(model, X_train, ['Net interest income', 'Intermediate income'], kind='both')

# 14.4.9 最优模型拟合效果图形展示
plt.figure()  # 创建另一个新的图形窗口
pred = model.predict(X_test)  # 对响应变量进行预测
t = np.arange(len(y_test))  # 求得响应变量在测试样本中的个数，以便绘制图形。
plt.plot(t, y_test, 'r-', linewidth=2, label='原值')  # 绘制响应变量原值曲线。
plt.plot(t, pred, 'g-', linewidth=2, label='预测值')  # 绘制响应变量预测曲线。
plt.legend(loc='upper right')  # 将图例放在图的右上方。
plt.grid()
plt.show()