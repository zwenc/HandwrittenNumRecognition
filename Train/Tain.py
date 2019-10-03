# -*- coding: utf-8 -*-
# @Time    : 2019/9/11 23:49
# @Author  : zwenc
# @File    : ClassNum2.py

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

original_TrainData = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

# 获得测试数据，本地没有就会通过网络下载
original_TestData = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,                                 # 下载测试集
    transform=torchvision.transforms.ToTensor(),  # 输出转换为Tensor类型
    download=True,                                # 如果本地没有，则下载
)

train_data = DataLoader(dataset=original_TrainData,batch_size = 20,shuffle = True)
# test_data = DataLoader(dataset=original_TestData,batch_size = 20,shuffle = True)

import matplotlib.pyplot as plt
for D,L in train_data:
    print(L[0])
    print(D[0][0])
    plt.imshow(D[0][0],cmap="gray")
    plt.show()

# 定义模块
model = Net().cuda()

# 定义损失函数，其实就是一个计算公式
criterion = nn.CrossEntropyLoss()

# 定义梯度下降算法,把model内的参数交给他
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum = 0.9)

for num_epoche in range(20):
    train_data = DataLoader(dataset=original_TrainData,batch_size = 20,shuffle = True) # 打乱数据，处理格式

    model.train(mode=True)  # 设置为训练模式

    for index, (data,lable) in enumerate(train_data):
        # data torch.Size([60, 1, 28, 28])
        D = torch.tensor(data,requires_grad=True).cuda()  # cuda表示放在GPU计算
        L = torch.tensor(lable).cuda()                    # cuda表示放在GPU计算

        out = model(D)
        loss = criterion(out,L)  # loss 是一个值，不是向量
        optimizer.zero_grad()    # 清除上一次的梯度，不然这次就会叠加
        loss.backward()          # 进行反向梯度计算
        optimizer.step()         # 更新参数

    model.eval()  # 设置网络为评估模式
    eval_loss = 0 # 保存平均损失
    num_count = 0 # 保存正确识别到的图片数量
    test_data = DataLoader(dataset=original_TestData,batch_size = 20,shuffle = True)
    for index, (data,lable) in enumerate(test_data):
        D = torch.tensor(data).cuda()
        L = torch.tensor(lable).cuda()

        out = model(D)
        loss = criterion(out,L)   # 计算损失，可以使用print输出
        eval_loss += loss.data.item() * L.size(0)  # loss.data.item()是mini-batch平均值

        pred = torch.max(out,1)[1] # 返回每一行中最大值的那个元素，且返回其索引。如果是0，则返回每列最大值
        num_count += (pred == L).sum() # 计算有多少个,这种方法只支持troch.tensor类型

    acc = num_count.float() / 10000
    eval_loss = eval_loss / 10000
    print("num_epoche:%2d，num_count:%5d, acc: %6.4f, eval_loss:%6.4f"%(num_epoche,num_count,acc,eval_loss))

    torch.save(model.state_dict(), "parameters.pt")



