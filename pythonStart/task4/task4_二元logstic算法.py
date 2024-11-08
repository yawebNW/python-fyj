# 二元Logistic回归算法

# 5.2.1 载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import RocCurveDisplay

# 2. 数据读取及观察
data = pd.read_csv('数据5.2.csv')
# 输出数据的信息，包括列数据类型、非空值数量等
print("数据信息:")
data.info()
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

# 3. 描述性分析
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# 计算各变量的统计指标（均值、标准差、最小值、最大值等）并输出
print("各变量统计指标:")
print(data.describe())

# 按照V1变量的取值分组对其他变量开展描述性分析并输出
print("按照V1分组的其他变量描述性分析:")
print(data.groupby('V1').describe().unstack())

# 针对分类变量“V1是否购买本次推广产品”“V3年收入水平”，使用交叉表的方式开展分析并输出
print("V1与V3的交叉表:")
print(pd.crosstab(data.V1, data.V3))
print("V1与V3的归一化交叉表:")
print(pd.crosstab(data.V1, data.V3, normalize='index'))

# 4. 数据处理
# 4.1 区分分类特征和连续特征并进行处理
def data_encoding(data):
    data = data[["V1", "V2", "V3", "V4", "V5", "V6", "V7"]]
    Discretefeature = ["V3"]
    Continuousfeature = ['V2', "V4", "V5", "V6", "V7"]
    df = pd.get_dummies(data, columns=Discretefeature)
    # 使用 for 循环将生成的独热编码列转换为整型
    for col in df.columns:
        if col.startswith('V3_'):  # 检查列名是否以 'V3_' 开头
            df[col] = df[col].astype(int)  # 将列转换为整型
    df[Continuousfeature] = (df[Continuousfeature] - df[Continuousfeature].mean()) / (df[Continuousfeature].std())
    df["V1"] = data[["V1"]]
    return df

data = data_encoding(data)

# 4.2 将样本示例全集分割为训练样本和测试样本
X = data.drop(['V1'], axis=1)  # 设置特征变量，即除V1之外的全部变量
# 为X增加1列，设置模型中的常数项（全为1）并输出X的前几行
print("添加常数项后的特征变量X前几行:")
X['intercept'] = [1] * X.shape[0]
y = data['V1']  # 设置响应变量，即V1
# 输出V1的取值计数
print("V1的取值计数:")
print(data["V1"].value_counts())
# 将数据分割为训练集和测试集，并输出训练集特征变量、测试集特征变量、训练集响应变量、测试集响应变量的前几行
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
print("训练集特征变量X_train前几行:")
print(X_train.head())
print("训练集响应变量y_train前几行:")
print(y_train.head())

# 5. 使用statsmodels建立二元Logistic回归算法模型
# 5.1 开展模型估计
model = sm.Logit(y_train, X_train)
results = model.fit()
# 输出模型的系数（未经指数转换）
print("模型系数（未经指数转换）:")
print(results.params)
# 输出模型的详细摘要信息
print("模型详细摘要:")
print(results.summary())
# 输出模型系数的指数形式（几率比）
print("模型系数的指数形式（几率比）:")
print(np.exp(results.params))
margeff = results.get_margeff()
# 输出边际效应的摘要信息
print("边际效应摘要:")
print(margeff.summary())

# 计算训练误差
table = results.pred_table()
# 输出预测结果表格（训练集）
print("训练集预测结果表格:")
print(table)

Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
# 输出训练准确率
print("训练准确率：", Accuracy)

Error_rate = 1 - Accuracy
# 输出训练误差率
print("训练误差率：", Error_rate)

precision = table[1, 1] / (table[0, 1] + table[1, 1])
# 输出训练精确率
print("训练精确率：", precision)

recall = table[1, 1] / (table[1, 0] + table[1, 1])
# 输出训练召回率
print("训练召回率：", recall)

# 计算测试误差
prob = results.predict(X_test)
pred = (prob >= 0.5)
table = pd.crosstab(y_test, pred, colnames=['Predicted'])
# 输出预测结果表格（测试集）
print("测试集预测结果表格:")
print(table)

table = np.array(table)

Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
# 输出测试准确率
print("测试准确率：", Accuracy)

Error_rate = 1 - Accuracy
# 输出测试误差率
print("测试误差率：", Error_rate)

precision = table[1, 1] / (table[0, 1] + table[1, 1])
# 输出测试精确率
print("测试精确率：", precision)

recall = table[1, 1] / (table[1, 0] + table[1, 1])
# 输出测试召回率
print("测试召回率：", recall)

# 5.2 使用sklearn建立二元Logistic回归算法模型
model = LogisticRegression(C=1e10, fit_intercept=True)
model.fit(X_train, y_train)
# 输出训练样本预测准确率
print("训练样本预测准确率: {:.3f}".format(model.score(X_train, y_train)))
# 输出测试样本预测准确率
print("测试样本预测准确率: {:.3f}".format(model.score(X_test, y_test)))
# 输出模型的系数
print("模型系数:")
print(model.coef_)

predict_target = model.predict(X_test)
# 输出测试集的预测类别
print("测试集预测类别:")
print(predict_target)
predict_target_prob = model.predict_proba(X_test)
# 输出测试集的预测概率
print("测试集预测概率:")
print(predict_target_prob)
predict_target_prob_lr = predict_target_prob[:, 1]
df = pd.DataFrame({'prob': predict_target_prob_lr, 'target': predict_target, 'labels': list(y_test)})
# 输出包含预测概率、预测类别和真实标签的DataFrame的前几行
print("包含预测概率、预测类别和真实标签的DataFrame前几行:")
print(df.head())

print('预测正确总数：')
print(sum(predict_target == y_test))

print('训练样本：')
predict_Target = model.predict(X_train)
# 输出训练样本的分类报告
print("训练样本分类报告:")
print(metrics.classification_report(y_train, predict_Target))
# 输出训练样本的混淆矩阵
print("训练样本混淆矩阵:")
print(metrics.confusion_matrix(y_train, predict_Target))

print('测试样本：')
# 输出测试样本的分类报告
print("测试样本分类报告:")
print(metrics.classification_report(y_test, predict_target))
# 输出测试样本的混淆矩阵
print("测试样本混淆矩阵:")
print(metrics.confusion_matrix(y_test, predict_target))

# 5.3 特征变量重要性水平分析
lr1 = [i for item in model.coef_ for i in item]
lr1 = np.array(lr1)
# 输出模型系数的一维数组形式
print("模型系数的一维数组形式:")
print(lr1)
feature = list(X.columns)
# 输出特征变量的列名列表
print("特征变量列名列表:")
print(feature)
dic = {}
for i in range(len(feature)):
    dic.update({feature[i]: lr1[i]})
# 输出特征变量与对应系数的字典
print("特征变量与对应系数的字典:")
print(dic)
df = pd.DataFrame.from_dict(dic, orient='index', columns=['权重'])
# 输出包含特征变量和对应权重的DataFrame
print("包含特征变量和对应权重的DataFrame:")
print(df)
df = df.reset_index().rename(columns={'index': '特征'})
# 输出重命名索引后的DataFrame（特征变量列名为'特征'，权重列名为'权重'）
print("重命名索引后的DataFrame:")
print(df)
df = df.sort_values(by='权重', ascending=False)
# 输出按照权重降序排序后的DataFrame
print("按照权重降序排序后的DataFrame:")
print(df)
data_hight = df['权重'].values.tolist()
data_x = df['特征'].values.tolist()

font = {'family': 'Times New Roman', 'size': 7, }
sns.set(font_scale=1.2)
plt.rc('font', family='Times New Roman')
plt.figure(figsize=(6, 6))
plt.barh(range(len(data_x)), data_hight, color='#6699CC')
plt.yticks(range(len(data_x)), data_x, fontsize=12)
plt.tick_params(labelsize=12)
plt.xlabel('Feature importance', fontsize=14)
plt.title("LR feature importance analysis", fontsize=14)
plt.show()

# 5.5.4 绘制ROC曲线，计算AUC值
RocCurveDisplay.from_estimator(model, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)
plt.title('ROC Curve (Test Set)')
plt.show()

# 5.5.5 计算科恩kappa得分
print("科恩kappa得分:")
print(cohen_kappa_score(y_test, pred))