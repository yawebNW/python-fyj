# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam

# 权重初始化函数，默认使用Xavier初始化
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:  # 排除不需要初始化的层（如embedding层）
            if len(w.size()) < 2:  # 跳过一维参数（如bias）
                continue
            if 'weight' in name:  # 初始化权重
                if method == 'xavier':
                    nn.init.xavier_normal_(w)  # Xavier初始化
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)  # Kaiming初始化
                else:
                    nn.init.normal_(w)  # 普通正态分布初始化
            elif 'bias' in name:  # 初始化偏置
                nn.init.constant_(w, 0)  # 偏置初始化为0
            else:
                pass

# 训练函数
def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()  # 记录训练开始时间
    model.train()  # 设置模型为训练模式
    param_optimizer = list(model.named_parameters())  # 获取模型参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # 不需要权重衰减的参数
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},  # 需要权重衰减的参数
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}  # 不需要权重衰减的参数
    ]
    # 使用BertAdam优化器
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录当前训练的batch数
    dev_best_loss = float('inf')  # 记录验证集上的最佳loss
    last_improve = 0  # 记录上次验证集loss下降的batch数
    model.train()  # 设置模型为训练模式
    for epoch in range(config.num_epochs):  # 遍历每个epoch
        flag = False  # 标记是否长时间没有提升
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):  # 遍历每个batch
            # 将数据移动到指定设备（GPU或CPU）
            trains = move_to_device(trains, config.device)
            labels = labels.to(config.device)
            outputs = model(trains)  # 前向传播
            model.zero_grad()  # 梯度清零
            loss = F.cross_entropy(outputs, labels).to(config.device)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            # 计算耗时
            time_dif = get_time_dif(start_time)
            # 在同一行输出 total_batch 和耗时
            print(f'\rIter: {total_batch:>6},  Time: {time_dif}', end='', flush=True)
            if total_batch % 100 == 0:  # 每100个batch输出一次训练和验证集的效果
                true = labels.data.cpu()  # 获取真实标签
                predic = torch.max(outputs.data, 1)[1].cpu()  # 获取预测标签
                train_acc = metrics.accuracy_score(true, predic)  # 计算训练集准确率
                dev_acc, dev_loss = evaluate(config, model, dev_iter)  # 计算验证集准确率和损失
                if dev_loss < dev_best_loss:  # 如果验证集loss下降
                    dev_best_loss = dev_loss  # 更新最佳loss
                    torch.save(model.state_dict(), config.save_path)  # 保存模型
                    improve = '*'  # 标记有提升
                    last_improve = total_batch  # 记录提升的batch数
                else:
                    improve = ''  # 无提升
                time_dif = get_time_dif(start_time)  # 计算训练时间
                msg = '\nIter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()  # 设置模型为训练模式
            total_batch += 1  # 更新batch数
            if (total_batch - last_improve > config.require_improvement):  # 如果长时间没有提升
                print("No optimization for a long time, auto-stopping...")  # 停止训练
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)  # 训练结束后进行测试

# 测试函数
def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))  # 加载最佳模型
    model.eval()  # 设置模型为评估模式
    start_time = time.time()  # 记录测试开始时间
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)  # 计算测试集效果
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)  # 打印分类报告
    print("Confusion Matrix...")
    print(test_confusion)  # 打印混淆矩阵
    time_dif = get_time_dif(start_time)  # 计算测试时间
    print("Time usage:", time_dif)

# 评估函数
def evaluate(config, model, data_iter, test=False):
    model.eval()  # 设置模型为评估模式
    loss_total = 0  # 记录总损失
    predict_all = np.array([], dtype=int)  # 记录所有预测结果
    labels_all = np.array([], dtype=int)  # 记录所有真实标签
    with torch.no_grad():  # 不计算梯度
        for texts, labels in data_iter:  # 遍历每个batch
            texts = move_to_device(texts, config.device)  # 将数据移动到指定设备
            labels = labels.to(config.device)
            outputs = model(texts)  # 前向传播
            loss = F.cross_entropy(outputs, labels)  # 计算损失
            loss_total += loss  # 累加损失
            labels = labels.data.cpu().numpy()  # 获取真实标签
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()  # 获取预测标签
            labels_all = np.append(labels_all, labels)  # 记录真实标签
            predict_all = np.append(predict_all, predic)  # 记录预测标签

    acc = metrics.accuracy_score(labels_all, predict_all)  # 计算准确率
    if test:  # 如果是测试集，返回分类报告和混淆矩阵
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)  # 返回准确率和平均损失

# 将数据移动到指定设备的函数
def move_to_device(data, device):
    """
    将输入数据（张量或元组）移动到指定设备。
    """
    if isinstance(data, tuple):  # 如果是元组，递归处理每个元素
        return tuple(move_to_device(t, device) for t in data)
    elif isinstance(data, torch.Tensor):  # 如果是张量，直接移动到设备
        return data.to(device)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")  # 不支持的数据类型