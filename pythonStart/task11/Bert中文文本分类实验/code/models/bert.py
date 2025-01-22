# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer  # 旧版BERT库
from pytorch_pretrained import BertModel, BertTokenizer  # 导入BERT模型和Tokenizer


# 配置类，用于定义模型的超参数和路径
class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'  # 模型名称
        self.train_path = dataset + '/data/train.txt'  # 训练集路径
        self.dev_path = dataset + '/data/dev.txt'  # 验证集路径
        self.test_path = dataset + '/data/test.txt'  # 测试集路径
        # 读取类别名单
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]
        # 模型保存路径
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        # 设备（GPU或CPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 100  # 若超过100个batch效果没有提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数量
        self.num_epochs = 3  # 训练的epoch数
        self.batch_size = 128  # 每个mini-batch的大小
        self.pad_size = 32  # 每句话处理成的长度（短填长切）
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain'  # 预训练BERT模型的路径
        # 加载BERT的Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768  # BERT模型的隐藏层大小


# 定义模型类
class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 设置BERT模型的参数为可训练
        for param in self.bert.parameters():
            param.requires_grad = True
            # 定义一个全连接层，将BERT的输出映射到类别数量
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

        # 定义前向传播过程

    def forward(self, x):
        context = x[0]  # 输入的句子（token IDs）
        mask = x[2]  # 对padding部分进行mask，padding部分用0表示
        # 通过BERT模型获取输出
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # 将BERT的输出通过全连接层进行分类
        out = self.fc(pooled)
        return out