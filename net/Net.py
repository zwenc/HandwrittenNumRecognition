# -*- coding: utf-8 -*-
# @Time    : 2019/9/12 17:49
# @Author  : zwenc
# @File    : CNNNet.py

import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        # 卷积层，输入一张图片，输出6张，滤波器为5*5大小，cuda表示使用GPU计算
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16 * 4 * 4,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    # 继承来自nn.Module的接口，必须实现，不能改名。
    # max_pool2d，池化函数，用来把图像缩小一半
    # relu 神经元激励函数，y = max(x,0)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))

        x = x.view(-1, 16 * 4 * 4) # 类似于reshape功能，重塑张量形状
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x