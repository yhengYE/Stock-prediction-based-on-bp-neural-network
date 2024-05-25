# -*- coding:utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import math
import torch
import torch.nn as nn
from model import NeuralNet
from torch.utils.data import Dataset, TensorDataset
from GUI import get_args,get_gui_args
import matplotlib.pyplot as plt

# 超参数
user_args = get_gui_args()
args = get_args()
args.batch_size = user_args['batch_size']
args.epochs = user_args['epochs']
args.learning_rate = user_args['learning_rate']
args.weight_decay = user_args['weight_decay']
args.activation_function = user_args['activation_function']
args.hidden_neurons_1=user_args['hidden_neurons_1']
args.hidden_neurons_2=user_args['hidden_neurons_2']
args.value=user_args['value']
'''-------------------------1、数据处理-------------------------'''

stockFile = 'CHRIS-MGEX_MW3.csv' #导入数据集
df = pd.read_csv(stockFile, index_col=0, parse_dates=[0])   # index_col:指定某列为索引  parse_dates:将某列解析为时间索引
data_last = df[args.value].values   # 将数据转为numpy.ndarray类型
data_last = data_last.tolist()  # 将数据转为list类型

# 切分训练集，测试集
x_train, y_train, x_test, y_test = [], [], [], []
temp_list = []


# 将前5600天为训练集数据，剩下的是测试集
for id, i in enumerate(data_last):
    if id+1 <= 5600:    # 前5600行是训练数据
        if (id+1) % 16 != 0:
            temp_list.append(i)
        else:
            x_train.append(temp_list)
            temp_list = []
            y_train.append([i])
    elif id+1 <= 6448:
        if (id + 1) % 16 != 0:
            temp_list.append(i)
        else:
            x_test.append(temp_list)
            temp_list = []
            y_test.append([i])

# 将数据转为 tensor类型
# 训练数据：15天一组，350组
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
# 测试数据：15天一组，53组
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)


'''-------------------------2、训练模型-------------------------'''


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 用gpu跑

# 制作dataset，dataloader
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)  # batch_size=args.batch_size指定了每个批次加载多少个样本
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# 损失函数
criterion = nn.MSELoss()  # 用于计算均方误差（Mean Squared Error, MSE）。它是回归问题（如股票价格预测）中常用的损失函数，用于衡量模型预测值与实际值之间的差异。
# 模型
model = NeuralNet(15,args.hidden_neurons_1, args.hidden_neurons_2, 1 , p=0, active_func=args.activation_function).to(device)

def init_weights(m):  # m为模块
    """初始化权重."""
    print(m)
    classname = m.__class__.__name__
    method = "normal_"
    if classname.find('Linear') != -1:
            m.weight.data.zero_()  # 先将定义线性层时的权重置为零
            nn.init.normal_(m.weight, mean=0, std=0.1)  # 正态分布初始化
            m.bias.data.zero_()


""" 定义优化器 """
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# 训练模型
epoch_train_loss = 0.0  # epoch 训练总损失
epoch_test_loss = 0.0  # epoch 测试总损失
total_train_length = len(train_loader)  # 训练集数量
total_test_length = len(test_loader)  # 测试集数量
test_label = []  # 保存测试集的标签
last_epoch_predict = []  # 最后一个批次预测结果
train_loss_list = []
test_loss_list = []
error_list = []
lr_list = []  # 学习率

# 初始化绘图
plt.ion()  # 开启interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 开始训练
for epoch in range(args.epochs):
    model.train()  # 训练模式
    error = 0.0  # 测试预测误差
    train_loss = 0.0  # 当前epoch的 训练损失
    test_loss = 0.0  # 当前epoch的 测试损失
    for idx, (x, y) in enumerate(train_loader):

        # gpu加速
        x = x.to(device)
        y = y.to(device)

        # 前向传播
        output = model(x)
        train_loss = criterion(output, y)  # 当前损失
        epoch_train_loss += train_loss.item()  # epoch总损失


        optimizer.zero_grad()  # 梯度清0
        train_loss.backward()  # 反向传播
        optimizer.step()  # 更新梯度

    with torch.no_grad():
        model.eval()  # 测试模式
        for idy, (x, y) in enumerate(test_loader):
            # gpu加速
            x = x.to(device)
            y = y.to(device)
            # 预测
            output = model(x)

            # predict_output += output.item()
            test_loss = criterion(output, y)  # 测试损失
            epoch_test_loss += test_loss.item()  # 计算 测试损失和
            error += math.fabs(y.item() - output.item())  # 误差计算公式：相减求绝对值

            # 保存最后一个批次预测结果，输出画图
            if epoch == args.epochs - 1:
                test_label.append(y.item())
                last_epoch_predict.append(output.item())

    epoch_train_loss /= total_train_length  # 计算epoch 平均训练损失
    epoch_test_loss /= total_test_length  # 计算epoch 平均测试损失
    error /= total_test_length  # MAE 评价均值误差
    train_loss_list.append(epoch_train_loss)  # 统计训练损失
    test_loss_list.append(epoch_train_loss)  # 统计测试损失
    error_list.append(error)  # 统计平均误差



    ax1.clear()
    ax1.plot(train_loss_list, label='Training Loss')
    ax1.plot(test_loss_list, label='Test Loss')
    ax1.set_title(f'Epoch {epoch + 1}/{args.epochs}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim([150, 2500])
    ax1.legend()

    ax2.clear()
    ax2.plot(error_list, label='MAE')
    ax2.set_title('MAE')
    ax2.set_title(f'Epoch {epoch + 1}/{args.epochs}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()

    plt.draw()  # 绘制最新的图
    plt.pause(0.1)  # 暂停一会儿，以便图表更新

plt.ioff()  # 关闭interactive mode

plt.show()
# 预测结果 画图
x = np.linspace(0, 53, 53)

# 损失图像
plt.subplot(3, 1, 1)
plt.plot(list(range(args.epochs)), train_loss_list, c="r", label="train_loss")
plt.plot(list(range(args.epochs)), test_loss_list, c="g", label="test_loss")
plt.legend()
# MAE 平均误差图像
plt.subplot(3, 1, 2)
plt.plot(error_list, c="r", label="MAE")
plt.text(len(error_list), error_list[-2] + 3, f"MAE={error_list[-2]}")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x, test_label, color='blue', marker='o')
plt.plot(x, last_epoch_predict, color='red', marker='o')
plt.xlabel('test data')
plt.ylabel('last price')
plt.legend(labels=['real', 'predict'])  # 加图例


plt.show()

