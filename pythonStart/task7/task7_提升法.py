# 第十五章 提升法
# 15.2.2 载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import RocCurveDisplay
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 15.4 二分类提升法示例
# 15.4.1 变量设置及数据处理
data = pd.read_csv('数据5.1.csv')
X = data.iloc[:, [1,5,6,7,8]]  # 设置特征变量
y = data.iloc[:, 0]  # 设置响应变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=10)

# 15.4.2 AdaBoost算法
model = AdaBoostClassifier(random_state=123)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"AdaBoost模型在测试集上的得分: {score:.3f}")

# 15.4.3 二分类提升法（默认参数）
model = GradientBoostingClassifier(random_state=123)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"二分类提升法模型在测试集上的得分: {score:.3f}")

# 15.4.4 使用随机搜索寻求最优参数
param_distributions = {'n_estimators': range(1, 300), 'max_depth': range(1, 10), 'subsample': np.linspace(0.1, 1, 10), 'learning_rate': np.linspace(0.1, 1, 10)}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=10), param_distributions=param_distributions, n_iter=10, cv=kfold, random_state=10,n_jobs=-1)
model.fit(X_train, y_train)
best_params = model.best_params_
model = model.best_estimator_
score = model.score(X_test, y_test)
print(f"最优参数: {best_params}，\n最优模型在测试集上的得分: {score:.3f}")

# 15.4.5 二分类问题提升法特征变量重要性水平分析
sorted_index = model.feature_importances_.argsort()
feature_importances = model.feature_importances_[sorted_index]
plt.barh(range(X_train.shape[1]), feature_importances)
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index])
plt.xlabel('特征变量重要性水平')
plt.ylabel('特征变量')
plt.title('二分类问题提升法特征变量重要性水平分析')
plt.tight_layout()
plt.show()

# 15.4.6 绘制部分依赖图与个体条件期望图
PartialDependenceDisplay.from_estimator(model, X_train, ['V2', 'V6'], kind='average')
PartialDependenceDisplay.from_estimator(model, X_train, ['V2', 'V6'], kind='individual')
PartialDependenceDisplay.from_estimator(model, X_train, ['V2', 'V6'], kind='both')
plt.figure()

# 15.4.7 模型性能评价
prob = model.predict_proba(X_test)
print(f"预测概率前5个样本:\n{prob[:5]}")
pred = model.predict(X_test)
print(f"预测结果前5个样本:\n{pred[:5]}")
print(f"混淆矩阵:\n{confusion_matrix(y_test, pred)}")
print(f"分类报告:\n{classification_report(y_test, pred)}")
print(f"Kappa得分: {cohen_kappa_score(y_test, pred):.3f}")

# 15.4.8 绘制ROC曲线
RocCurveDisplay.from_estimator(model, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)
plt.title('二分类问题提升法ROC曲线')
plt.show()

# 15.4.9 运用两个特征变量绘制二分类提升法决策边界图
X2 = X.iloc[:, [0, 1]]  # 仅选择资产负债率、主营业务收入作为特征变量
model = GradientBoostingClassifier(random_state=123)
model.fit(X2, y)
score = model.score(X2, y)
print(f"模型预测准确率：{score:.3f}")
plot_decision_regions(np.array(X2), np.array(y), model)
plt.xlabel('资产负债率')
plt.ylabel('主营业务收入')
plt.title('二分类问题提升法决策边界')
plt.show()

# 15.4.10 XGBoost二分类提升法
model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, max_depth=6, subsample=0.6, colsample_bytree=0.8, learning_rate=0.1, random_state=0)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"XGBoost二分类提升法模型在测试集上的得分: {score:.3f}")
