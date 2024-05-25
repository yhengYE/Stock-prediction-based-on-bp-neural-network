# -*- coding:utf-8 -*-
import torch.nn as nn
import torch

class Swish(nn.Module):


    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, p=0, active_func="relu"):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(p)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        # 根据传入的激活函数名称选择激活函数
        if active_func.lower() == "relu":
            self.active_func1 = nn.ReLU()
            self.active_func2 = nn.ReLU()
        elif active_func.lower() == "swish":
            self.active_func1 = Swish()
            self.active_func2 = Swish()



    def forward(self, x):
        out = self.fc1(x)
        out = self.active_func1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.active_func2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out
    
