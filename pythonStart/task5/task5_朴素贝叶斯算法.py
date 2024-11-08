#第八章  朴素贝叶斯算法
# 2.2  载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_decision_regions

# 1、载入分析所需要的库和模块
# 代码中已经在开头部分完成了库和模块的导入

# 2、数据读取及观察
data = pd.read_csv('数据5.1.csv')
print("数据信息：")
print(data.info())
print("\n是否有空值？")
print(data.isnull().values.any())
print("\n'V1征信违约记录'值计数：")
print(data['V1'].value_counts())
print("\n'V1征信违约记录'值计数占比：")
print(data['V1'].value_counts(normalize=True))

# 3、将样本示例全集分割为训练样本和测试样本
# 选择特定特征变量
X = data[['V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9']]
y = data['V1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

# 4、构建高斯朴素贝叶斯算法模型
# 4.1 高斯朴素贝叶斯算法拟合
model_gaussian = GaussianNB()
model_gaussian.fit(X_train, y_train)
print("\n高斯朴素贝叶斯算法在测试集上的得分：")
print(model_gaussian.score(X_test, y_test))

# 4.2 绘制ROC曲线
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决图表中中文显示问题
RocCurveDisplay.from_estimator(model_gaussian, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)
plt.title('高斯朴素贝叶斯ROC曲线')
plt.show()

# 4.3 只运用“V7利息保障倍数”“V8银行负债”两个特征变量开展高斯朴素贝叶斯算法，并绘制高斯朴素贝叶斯决策边界图
X2 = X[['V7', 'V8']]
model_gaussian = GaussianNB()
model_gaussian.fit(X2, y)
print("\n使用两个特征变量的高斯朴素贝叶斯算法在测试集上的得分：")
print(model_gaussian.score(X2, y))
plt.rcParams['axes.unicode_minus'] = False  # 解决图表中负号不显示问题。
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决图表中中文显示问题
plot_decision_regions(np.array(X2), np.array(y), model_gaussian)
plt.xlabel('利息保障倍数')  # 将 x 轴设置为'利息保障倍数'
plt.ylabel('银行负债')  # 将 y 轴设置为'银行负债'
plt.title('高斯朴素贝叶斯决策边界')  # 将标题设置为'高斯朴素贝叶斯决策边界'
plt.show()

# 5、构建多项式朴素贝叶斯算法模型
model_multinomial = MultinomialNB(alpha=0)  # 不进行拉普拉斯修正
model_multinomial.fit(X_train, y_train)
print("\n多项式朴素贝叶斯算法（不进行拉普拉斯修正）在测试集上的得分：")
print(model_multinomial.score(X_test, y_test))

model_multinomial = MultinomialNB(alpha=1)  # 进行拉普拉斯修正
model_multinomial.fit(X_train, y_train)
print("\n多项式朴素贝叶斯算法（进行拉普拉斯修正）在测试集上的得分：")
print(model_multinomial.score(X_test, y_test))

# 6、构建补集朴素贝叶斯算法模型
model_complement = ComplementNB(alpha=1)  # 进行拉普拉斯修正
model_complement.fit(X_train, y_train)
print("\n补集朴素贝叶斯算法在测试集上的得分：")
print(model_complement.score(X_test, y_test))

# 7、构建二项式朴素贝叶斯算法模型
model_bernoulli = BernoulliNB(alpha=1)  # 进行拉普拉斯修正
model_bernoulli.fit(X_train, y_train)
print("\n二项式朴素贝叶斯算法在测试集上的得分：")
print(model_bernoulli.score(X_test, y_test))

model_bernoulli = BernoulliNB(binarize=2, alpha=1)  # 设置参数 binarize=2，进行拉普拉斯修正
model_bernoulli.fit(X_train, y_train)
print("\n二项式朴素贝叶斯算法（设置 binarize=2）在测试集上的得分：")
print(model_bernoulli.score(X_test, y_test))

# 8、寻求二项式朴素贝叶斯算法拟合的最优参数
# 8.1 通过将样本分割为训练样本、验证样本、测试样本的方式寻找最优参数
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=10)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=100)
print("\n训练集形状：", y_train.shape)
print("验证集形状：", y_val.shape)
print("测试集形状：", y_test.shape)

best_val_score = 0
for binarize in np.arange(0.5, 5.5, 0.5):
    for alpha in np.arange(0.1, 1.1, 0.1):
        model = BernoulliNB(binarize=binarize, alpha=alpha)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        if score > best_val_score:
            best_val_score = score
            best_val_parameters = {'binarize': binarize, 'alpha': alpha}

print("\n验证集的最优预测准确率：")
print(best_val_score)
print("\n最优参数：")
print(best_val_parameters)

model = BernoulliNB(**best_val_parameters)
model.fit(X_trainval, y_trainval)
print("\n使用最优参数的二项式朴素贝叶斯算法在测试集上的得分：")
print(model.score(X_test, y_test))

# 8.2 采用 10 折交叉验证方法寻找最优参数
param_grid = {'binarize': np.arange(0.5, 5.5, 0.5), 'alpha': np.arange(0.1, 1.1, 0.1)}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(BernoulliNB(), param_grid, cv=kfold)
model.fit(X_trainval, y_trainval)
print("\n使用 10 折交叉验证的二项式朴素贝叶斯算法在测试集上的得分：")
print(model.score(X_test, y_test))
print("\n最优参数：")
print(model.best_params_)
print("\n最优预测准确率：")
print(model.best_score_)

outputs = pd.DataFrame(model.cv_results_)
pd.set_option('display.max_columns', None)
print("\n前 3 行交叉验证信息：")
print(outputs.head(3))

scores = np.array(outputs.mean_test_score).reshape(10, 10)
ax = sns.heatmap(scores, cmap='Oranges', annot=True, fmt='.3f')
ax.set_xlabel('binarize')
ax.set_xticklabels([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
ax.set_ylabel('alpha')
ax.set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.tight_layout()
plt.show()

# 9、开展最优二项式朴素贝叶斯算法模型性能评价
prob = model.predict_proba(X_test)
print("\n前几个预测概率：")
print(prob[:5])

pred = model.predict(X_test)
print("\n前几个预测结果：")
print(pred[:5])

print("\n测试集的混淆矩阵：")
print(confusion_matrix(y_test, pred))
print("\n分类报告：")
print(classification_report(y_test, pred))
print("\nKappa 得分：")
print(cohen_kappa_score(y_test, pred))