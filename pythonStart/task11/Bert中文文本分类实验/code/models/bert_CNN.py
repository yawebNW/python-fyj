# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer  # 导入BERT模型和Tokenizer


class Config(object):
    """配置参数类，用于定义模型的超参数和路径"""

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

        self.require_improvement = 20  # 若超过100个batch效果没有提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数量
        self.num_epochs = 3  # 训练的epoch数
        self.batch_size = 128  # 每个mini-batch的大小
        self.pad_size = 32  # 每句话处理成的长度（短填长切）
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain'  # 预训练BERT模型的路径
        # 加载BERT的Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768  # BERT模型的隐藏层大小
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 1693  # 卷积核数量（channels数）
        self.dropout = 0.1  # Dropout概率


class Model(nn.Module):
    """BERT-CNN模型类"""

    def __init__(self, config):
        super(Model, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 设置BERT模型的参数为可训练
        for param in self.bert.parameters():
            param.requires_grad = True
        # 定义多个卷积层，每个卷积层的输入通道为1，输出通道为num_filters，卷积核尺寸为(k, hidden_size)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        # 定义Dropout层
        self.dropout = nn.Dropout(config.dropout)
        # 定义全连接层，将卷积层的输出映射到类别数量
        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        # 将模型移动到指定设备（GPU或CPU）
        self.to(config.device)

    def conv_and_pool(self, x, conv):
        """卷积和池化操作"""
        x = F.relu(conv(x)).squeeze(3)  # 对输入进行卷积和ReLU激活，然后去掉最后一个维度
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # 对卷积结果进行最大池化，然后去掉最后一个维度
        return x

    def forward(self, x):
        """前向传播过程"""
        context = x[0]  # 输入的句子（token IDs）
        mask = x[2]  # 对padding部分进行mask，padding部分用0表示
        # 通过BERT模型获取输出
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # 将BERT的输出增加一个维度，作为卷积层的输入
        out = encoder_out.unsqueeze(1)
        # 对BERT的输出进行卷积和池化操作，并将结果拼接在一起
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # 对卷积结果进行Dropout
        out = self.dropout(out)
        # 将Dropout后的结果通过全连接层进行分类
        out = self.fc_cnn(out)
        return out