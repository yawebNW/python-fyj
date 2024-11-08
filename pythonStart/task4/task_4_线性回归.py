# 1  载入分析所需要的模块和函数
import pandas as pd  # 载入pandas模块，并简称为pd
import numpy as np  # 载入numpy模块，并简称为np
import matplotlib.pyplot as plt  # 载入matplotlib.pyplot模块，并简称为plt
import seaborn as sns  # 载入seaborn模块，并简称为sns
from scipy import stats  # 载入stats模块
from scipy.stats import probplot  # 载入probplot模块
import statsmodels.formula.api as smf  # 载入statsmodels.formula.api模块，并简称为smf
from sklearn.linear_model import LinearRegression  # 载入LinearRegression模块
from sklearn.model_selection import train_test_split  # 载入train_test_split模块
from sklearn.metrics import mean_squared_error, r2_score  # 载入mean_squared_error, r2_score模块

data = pd.read_csv('数据4.3.csv')
print('Profit contribution数据的描述性分析:')
print(data['Profit contribution'].describe())

print('Net interest income数据的描述性分析:')
print(data['Net interest income'].describe())

print('Intermediate income数据的描述性分析:')
print(data['Intermediate income'].describe())

print('Deposit and finance daily数据的描述性分析:')
print(data['Deposit and finance daily'].describe())

# Shapiro-Wilk test检验
Ho = '数据服从正态分布'  # 定义原假设
Ha = '数据不服从正态分布'  # 定义备择假设
alpha = 0.05  # 定义显著性P值


def normality_check_shapiro(data, columnName):
    print("Shapiro test for {columnName}".format(columnName=columnName))
    res = stats.shapiro(data[columnName])
    pValue = round(res[1], 3)
    if pValue > alpha:
        print("pvalue = {pValue} > {alpha}. 不能拒绝原假设. {Ho}".format(pValue=pValue, alpha=alpha, Ho=Ho))
    else:
        print("pvalue = {pValue} <= {alpha}. 拒绝原假设. {Ha}".format(pValue=pValue, alpha=alpha, Ha=Ha))


print('进行Shapiro-Wilk test 检验')
normality_check_shapiro(data, data.columns[1])
normality_check_shapiro(data, data.columns[2])
normality_check_shapiro(data, data.columns[3])
normality_check_shapiro(data, data.columns[4])


# kstest检验
def normality_check_ks(data, columnName):
    print("kstest for {columnName}".format(columnName=columnName))
    res = stats.kstest(data[columnName], 'norm')
    pValue = round(res[1], 3)
    if pValue > alpha:
        print("pvalue = {pValue} > {alpha}. 不能拒绝原假设. {Ho}".format(pValue=pValue, alpha=alpha, Ho=Ho))
    else:
        print("pvalue = {pValue} <= {alpha}. 拒绝原假设. {Ha}".format(pValue=pValue, alpha=alpha, Ha=Ha))


print('进行kstest 检验')
normality_check_ks(data, data.columns[1])
normality_check_ks(data, data.columns[2])
normality_check_ks(data, data.columns[3])
normality_check_ks(data, data.columns[4])

#相关性分析
part_data = data.iloc[:,[1,2,3,4]]
print('皮尔逊相关系数矩阵如下:')
print(part_data.corr(method='pearson').to_string()) #输出变量之间的皮尔逊相关系数矩阵
plt.subplot(1,1,1)
sns.heatmap(part_data.corr(), annot=True)# 绘制相关矩阵的热图
plt.show()
print('斯皮尔曼相关系数矩阵如下:')
print(part_data.corr(method='spearman').to_string()) #输出变量之间的斯皮尔曼等级相关系数矩阵
print('肯德尔相关系数矩阵如下:')
print(part_data.corr(method='kendall').to_string()) #输出变量之间的肯德尔等级相关系数矩阵

#7  使用 smf 进行线性回归
X = data.iloc[:, 2:5]#将数据集中的第3列至第5列作为自变量
y = data.iloc[:, 1:2] #将数据集中的第2列作为因变量
model = smf.ols('y~X', data=data).fit()#使用线性回归模型，并进行训练
print(model.summary())#输出估计模型摘要
y_pred = model.predict(X)#将y_pred设定为模型因变量的预测值
y_lst = y.iloc[:, -1:].values.tolist()#将y_lst设定为模型因变量的实际值，并转换为列表形式
plt.scatter( y_lst, y_pred, color='blue')  # 绘制因变量实际值和拟合值的散点图
plt.plot( y_lst, y_pred, color='Red', linewidth=2 )  # 绘制因变量实际值和拟合值的的线图
plt.title('Simple Linear Regression')#设定图片标题为Simple Linear Regression
plt.xlabel('y')#设定x轴为y
plt.ylabel('y_pred')#设定y轴为y_pred
plt.show()
#7.2  多重共线性检验
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print('多重共线性检验结果如下')
print(vif.round(1))

#8  使用 sklearn 进行线性回归
#8.1  使用验证集法进行模型拟合
X = data.iloc[:, 2:5]#将数据集中的第4列至第5列作为自变量
y = data.iloc[:, 1:2] #将数据集中的第2列作为因变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)#将样本示例全集划分为训练样本和测试样本，测试样本占比为30%。
print('四个数据形状如下:')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)#观察四个数据的形状
model = LinearRegression()#使用线性回归模型
model.fit(X_train, y_train)#基于训练样本拟合模型
print('估计的回归系数值为:')
print(model.coef_)#计算上步估计得到的回归系数值
model.score(X_test, y_test)#观察模型在测试集中的拟合优度（可决系数）
pred = model.predict(X_test)#计算响应变量基于测试集的预测结果
print('数据形状:')
print(pred.shape)#观察数据形状
print('测试集的均方误差:')
print(mean_squared_error(y_test, pred))#计算测试集的均方误差
print('测试集的可决系数:')
print(r2_score(y_test, pred))#计算测试集的可决系数

#8.2  更换随机数种子，使用验证集法进行模型拟合
print('更换随机数种子')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)#更换随机数种子
model = LinearRegression().fit(X_train, y_train)#基于训练样本拟合线性回归模型
pred = model.predict(X_test)#计算响应变量基于测试集的预测结果
print('此时测试集的均方误差:')
print(mean_squared_error(y_test, pred))#计算测试集的均方误差
print('测试集的可决系数:')
print(r2_score(y_test, pred))#计算测试集的可决系数

# 8.3  使用10折交叉验证法进行模型拟合
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedKFold
print('使用10折交叉验证法')
X = data.iloc[:, 2:5]#将数据集中的第4列至第5列作为自变量
y = data.iloc[:, 1:2] #将数据集中的第2列作为因变量
model = LinearRegression()#使用线性回归模型
kfold = KFold(n_splits=10,shuffle=True, random_state=1)#将样本示例全集分为10折
scores = cross_val_score(model, X, y, cv=kfold)#计算每一折的可决系数
print('每一折的可决系数:')
print(scores)#显示每一折的可决系数
print('各折样本可决系数的均值:%f'%scores.mean())#计算各折样本可决系数的均值
print('各折样本可决系数的标准差:%f'%scores.std())#计算各折样本可决系数的标准差
scores_mse = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')#得到每个子样本的均方误差
print('各折样本的均方误差:')
print(scores_mse)#显示各折样本的均方误差
print('各折样本均方误差的均值:%f'%scores_mse.mean())#计算各折样本均方误差的均值
# 更换随机数种子，并与上步得到结果进行对比，观察均方误差MSE大小
kfold = KFold(n_splits=10, shuffle=True, random_state=100)
scores_mse = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')#得到每个子样本的均方误差
print('更换随机数后各折样本均方误差的均值:%f'%scores_mse.mean())#计算各折样本均方误差的均值

#8.4  使用10折重复10次交叉验证法进行模型拟合
print('使用10折重复10次交叉验证法进行模型拟合')
rkfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
scores_mse = -cross_val_score(model, X, y, cv=rkfold, scoring='neg_mean_squared_error')#得到每个子样本的均方误差
print("均方误差的形状:%f"%scores_mse.shape)
print('各个子样本均方误差的均值:%f'%scores_mse.mean())
# 绘制各子样本均方误差的直方图
sns.displot(pd.DataFrame(scores_mse))
plt.xlabel('MSE')
plt.title('10-fold CV Repeated 10 Times')
plt.show()

# 8.5  使用留一交叉验证法进行模型拟合
loo = LeaveOneOut()
scores_mse = -cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
print('使用留一交叉验证法进行模型拟合')
print('样本均方误差的均值:%f'%scores_mse.mean())
