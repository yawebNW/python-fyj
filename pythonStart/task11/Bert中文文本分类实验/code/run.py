# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 使用的数据集名称

    model_name = args.model  # 从命令行参数获取模型名称，例如 'Bert'
    x = import_module('models.' + model_name)  # 动态导入对应的模型模块
    config = x.Config(dataset)  # 初始化模型的配置

    # 设置随机种子以确保实验的可重复性
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 确保每次运行的结果一致

    # 检查CUDA是否可用，并打印当前设备信息
    print("CUDA Available:", torch.cuda.is_available())
    print("Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")

    start_time = time.time()  # 记录开始时间
    print("Loading data...")
    # 构建训练集、验证集和测试集
    train_data, dev_data, test_data = build_dataset(config)
    # 构建数据迭代器
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)  # 计算数据加载所用时间
    print("Time usage:", time_dif)

    # 初始化模型并将其移动到指定的设备（GPU或CPU）
    model = x.Model(config).to(config.device)
    # 开始训练模型
    train(config, model, train_iter, dev_iter, test_iter)