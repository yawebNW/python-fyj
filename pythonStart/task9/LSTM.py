# 首先 import 一些主要的包
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
# 加载 pytorch
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# 简单读出一个股票
# 获取文件名
file_name = 'train_data.npy'
# 读取数组
data = np.load(file_name)
# 简单展示信息
print(data)
# 新建一个图像
plt.figure(figsize=(20, 10))
# 绘画该股票不同的时间段的图像
plt.rcParams['font.family'] = 'Microsoft YaHei'  # 替换为你选择的字体
plt.plot(data, c='blue')
plt.xlabel("时间")
plt.ylabel("股价")
plt.title("某股票走势图")
# 展示图像
plt.legend()
plt.show()
# 这个 [0, 300] 是手动的预设值，可以自己更改
scaler = MinMaxScaler().fit(data.reshape(-1, 1))

# 生成题目所需的训练集合
def generate_data(data):
    # 记录 data 的长度
    n = data.shape[0]
    # 目标是生成可直接用于训练和测试的 x 和 y
    x = []
    y = []
    # 建立 (14 -> 1) 的 x 和 y
    for i in range(15, n):
        x.append(data[i - 15:i - 1])
        y.append(data[i])
    # 转换为 numpy 数组
    x = np.array(x)
    y = np.array(y)
    return x, y

x, y = generate_data(data)
print('x.shape : ', x.shape)
print('y.shape : ', y.shape)

# 生成 train valid test 集合，以供训练所需
def generate_training_data(x, y):
    # 样本总数
    num_samples = x.shape[0]
    # 测试集大小
    num_test = round(num_samples * 0.2)
    # 训练集大小
    num_train = round(num_samples * 0.7)
    # 校验集大小
    num_val = num_samples - num_test - num_train
    # 训练集拥有从 0 起长度为 num_train 的样本
    x_train, y_train = x[:num_train], y[:num_train]
    # 校验集拥有从 num_train 起长度为 num_val 的样本
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # 测试集拥有尾部 num_test 个样本
    x_test, y_test = x[-num_test:], y[-num_test:]
    # 返回这些集合
    return x_train, y_train, x_val, y_val, x_test, y_test

# 获取数据中的 x, y
x, y = generate_data(data)
# 将 x,y 转换乘 tensor ， Pytorch 模型默认的类型是 float32
x = torch.tensor(x)
y = torch.tensor(y)
print(x.shape, y.shape)
# 将 y 转化形状
y = y.view(y.shape[0], 1)
print(x.shape, y.shape)
# 对 x, y 进行 minmaxscale
x_scaled = scaler.transform(x.reshape(-1, 1)).reshape(-1, 14)
y_scaled = scaler.transform(y)
x_scaled = torch.tensor(x_scaled, dtype=torch.float32)
y_scaled = torch.tensor(y_scaled, dtype=torch.float32)
# 处理出训练集，校验集和测试集
x_train, y_train, x_val, y_val, x_test, y_test = generate_training_data(x_scaled, y_scaled)

# 建立一个自定 Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


# 建立训练数据集、校验数据集和测试数据集
train_data = MyDataset(x_train, y_train)
valid_data = MyDataset(x_val, y_val)
test_data = MyDataset(x_test, y_test)

# 规定批次的大小
batch_size = 512

# 创建对应的 DataLoader
train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 校验集和测试集的 shuffle 是没有必要的，因为每次都会全部跑一遍
valid_iter = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_iter = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
# for i, read_data in enumerate(test_iter):
#     # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
#     print("第 {} 个Batch \n{}".format(i, read_data))
#     break
# # 表示输出数据
# print(read_data[0].shape, read_data[0])
# # 表示输出标签
# print(read_data[1].shape, read_data[1])

# 输入的数量是前 14 个交易日的收盘价
num_inputs = 14
# 输出是下一个交易日的收盘价
num_outputs = 1
# 隐藏层的个数
num_hiddens = 128

# 建立一个稍微复杂的 LSTM 模型
class LSTM(nn.Module):
    def __init__(self, num_hiddens, num_outputs):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=num_hiddens,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(num_hiddens, num_outputs)

    def forward(self, x):
        x = x.view(x.shape[0], -1, 1)
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.fc(r_out[:, -1, :])  # 只需要最后一个的output
        return out

# 判断 gpu 是否可用
use_gpu = torch.cuda.is_available()

def compute_mae(y_hat, y):
    '''
    :param y: 标准值
    :param y_hat: 用户的预测值
    :return: MAE 平均绝对误差 mean(|y*-y|)
    '''
    return torch.mean(torch.abs(y_hat - y))

def compute_mape(y_hat, y):
    '''
    :param y: 标准值
    :param y_hat: 用户的预测值
    :return: MAPE 平均百分比误差 mean(|y*-y|/y)
    '''
    return torch.mean(torch.abs(y_hat - y) / y)

def evaluate_accuracy(data_iter, model):
    '''
    :param data_iter: 输入的 DataLoader
    :param model: 用户的模型
    :return: 对应的 MAE 和 MAPE
    '''
    # 初始化参数
    mae_sum, mape_sum, n = 0.0, 0.0, 0
    # 对每一个 data_iter 的每一个 x,y 进行计算
    for x, y in data_iter:
        # 如果运行在 GPU 上，需要将内存中的 x 拷贝到显存中
        if (use_gpu):
            x = x.cuda()
        # 计算模型得出的 y_hat
        y_hat = model(x)
        # 将 y_hat 逆归一化，这里逆归一化需要将数据转移到 CPU 才可以进行
        y_hat_real = torch.from_numpy(
            scaler.inverse_transform(np.array(y_hat.detach().cpu()).reshape(-1, 1)).reshape(y_hat.shape))
        y_real = torch.from_numpy(scaler.inverse_transform(np.array(y.reshape(-1, 1))).reshape(y.shape))
        # 计算对应的 MAE 和 RMSE 对应的和，并乘以 batch 大小
        mae_sum += compute_mae(y_hat_real, y_real) * y.shape[0]
        mape_sum += compute_mape(y_hat_real, y_real) * y.shape[0]
        # n 用于统计 DataLoader 中一共有多少数量
        n += y.shape[0]
    # 返回时需要除以 batch 大小，得到平均值
    return mae_sum / n, mape_sum / n

# 使用均方根误差
loss = torch.nn.MSELoss()

# 自定义的损失函数，可以直接调用
def my_loss_func(y_hat, y):
    return compute_mae(y_hat, y)

# 使用上面描述的线性网络
# model = LinearNet(num_inputs, num_outputs)
model = LSTM(num_hiddens, num_outputs)
# 使用 Adam 优化器， learning rate 调至 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 也可选用 SGD 或其他优化器
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1)
def train_model(model, train_iter, valid_iter, loss, num_epochs,
                params=None, optimizer=None):
    # 用于绘图用的信息
    train_losses, valid_losses, train_maes, train_mapes, valid_maes, valid_mapes = [], [], [], [], [], []
    # 循环 num_epochs 次
    for epoch in range(num_epochs):
        # 初始化参数
        train_l_sum, n = 0.0, 0
        # 初始化时间
        start = time.time()
        # 模型改为训练状态，如果使用了 dropout, batchnorm 之类的层时，训练状态和评估状态的表现会有巨大差别
        model.train()
        # 对训练数据集的每个 batch 执行
        for x, y in train_iter:
            # 如果使用了 GPU 则拷贝进显存
            if (use_gpu):
                x, y = x.cuda(), y.cuda()
            # 计算 y_hat
            y_hat = model(x)
            # 计算损失
            l = loss(y_hat, y).mean()
            # 梯度清零
            optimizer.zero_grad()
            # L1 正则化
            # for param in params:
            #     l += torch.sum(torch.abs(param))
            # L2 正则化可以在 optimizer 上加入 weight_decay 的方式加入
            # 求好对应的梯度
            l.backward()
            # 执行一次反向传播
            optimizer.step()
            # 对 loss 求和（在下面打印出来）
            train_l_sum += l.item() * y.shape[0]
            # 计数一共有多少个元素
            n += y.shape[0]
        # 模型开启预测状态
        model.eval()
        # 同样的，我们可以计算验证集上的 loss
        valid_l_sum, valid_n = 0, 0
        for x, y in valid_iter:
            # 如果使用了 GPU 则拷贝进显存
            if (use_gpu):
                x, y = x.cuda(), y.cuda()
            # 计算 y_hat
            y_hat = model(x)
            # 计算损失
            l = loss(y_hat, y).mean()
            # 对 loss 求和（在下面打印出来）
            valid_l_sum += l.item() * y.shape[0]
            # 计数一共有多少个元素
            valid_n += y.shape[0]
        # 对验证集合求指标
        # 这里训练集其实可以在循环内高效地直接算出，这里为了代码的可读性牺牲了效率
        train_mae, train_mape = evaluate_accuracy(train_iter, model)
        valid_mae, valid_mape = evaluate_accuracy(valid_iter, model)
        if (epoch + 1) % 100 == 0:
            print(
                'epoch %d, train loss %.6f, valid loss %.6f, train mae %.6f, mape %.6f, valid mae %.6f,mape %.6f, time %.2f sec'
                % (epoch + 1, train_l_sum / n, valid_l_sum / valid_n, train_mae, train_mape, valid_mae, valid_mape,
                   time.time() - start))
        # 记录绘图有关的信息
        train_losses.append(train_l_sum / n)
        valid_losses.append(valid_l_sum / valid_n)
        train_maes.append(train_mae)
        train_mapes.append(train_mape)
        valid_maes.append(valid_mae)
        valid_mapes.append(valid_mape)
    # 返回一个训练好的模型和用于绘图的集合
    return model, (train_losses, valid_losses, train_maes, train_mapes, valid_maes, valid_mapes)

# 训练模型
model, (train_losses, valid_losses, train_maes, train_mapes, valid_maes, valid_mapes) = (
    train_model(model, train_iter, test_iter, loss, 1000, model.parameters(), optimizer))
# 为了方便储存与读取，建立成一个元组
draw_data = (train_losses, valid_losses, train_maes, train_mapes, valid_maes, valid_mapes)

# 新建一个图像
plt.figure(figsize=(16, 8))
# 绘制 train_loss 曲线
plt.plot(train_losses, label='train_loss')
# 绘制 valid_loss 曲线
plt.plot(valid_losses, label='valid_loss')
plt.xlabel("训练次数")
plt.ylabel("loss")
plt.title("loss图")
# 展示带标签的图像
plt.legend()
plt.show()

# 新建一个图像
plt.figure(figsize=(16, 8))
# 绘画结点
plt.plot(train_maes, c='blue', label='train_mae')
plt.plot(train_mapes, c='red', label='train_rmse')
plt.plot(valid_maes, c='green', label='valid_mae')
plt.plot(valid_mapes, c='orange', label='valid_rmse')
plt.xlabel("训练次数")
plt.ylabel("mae及rmse")
plt.title("mae图")
# 展示图像
plt.legend()
plt.show()

# 新建一个图像
plt.figure(figsize=(16, 8))
# 预测结果
y_hat = model(x_test)
# 取前 300 个测试集
num_for_draw = y_test.shape[0]
# 绘画某些结点第一天的情况
plt.plot(scaler.inverse_transform(y_test[:num_for_draw].reshape(-1, 1)).reshape(-1), c='blue', label='y_test')
plt.plot(scaler.inverse_transform(y_hat[:num_for_draw].detach().numpy().reshape(-1, 1)).reshape(-1), c='red',label='y_hat')
plt.xlabel("时间")
plt.ylabel("股价")
plt.title("预测对比图")
# 展示图像
plt.legend()
plt.show()
# 获得测试集的数据
test_mae, test_mape = evaluate_accuracy(test_iter, model)
print('test mae, rmse: %.3f,%.3f' % (test_mae, test_mape))
