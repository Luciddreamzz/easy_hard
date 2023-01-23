#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from load_demo_img import get_input_img
def load_data(batch_size):
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # 使用Compose对象组装多个变换：转为tensor，标准化
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)  # 指定训练集
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)  # 指定测试集

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)  # 训练集加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试集加载器
    return train_loader, test_loader


def draw_pic(index, cost_data):
    plt.ion()
    plt.plot(index, cost_data,color='dodgerblue', marker='')
    plt.title("cost figure")
    plt.xlabel('item_time')
    plt.ylabel('cost')
    plt.show() 
    plt.pause(0.000001)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # sequential用于构建序列型神经网络

        self.conv1 = nn.Sequential(nn.Conv2d(1,16, kernel_size=5, stride=1, padding=2),nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(True))
        self.pool = nn.MaxPool2d(2, 2)
        self.FC1 = nn.Sequential(nn.Linear(32*7*7, 64), nn.ReLU(True))
        self.FC2 = nn.Sequential(nn.Linear(64, 48), nn.ReLU(True))
        self.FC3 = nn.Sequential(nn.Linear(48, 10))

    def forward(self, x):
        x = self.conv1(x)

        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)

        x = x.view(-1, 32*7*7)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        return x

    @staticmethod
    def check_opt(opt, lr):
        if opt == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=lr)
        elif opt == "Adam":
            optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-08, betas=(0.9, 0.999))
        elif opt == 'Adagrad':
            optimizer = optim.Adagrad(net.parameters(), lr=lr, lr_decay=0.5)
        else:
            raise ValueError('You Enter A Wrong Optimizer!')
        return optimizer

    def train(self, epoch_num=5, lr=0.001, criterion=nn.CrossEntropyLoss(), opt='SGD'):
        x_label = []
        losses = []
        j = 0
        start_time = time.time()
        optimizer = self.check_opt(opt, lr)
        for i in range(epoch_num):
            for data, label in train_loader:
                data = torch.autograd.Variable(data)
                label = torch.autograd.Variable(label)
                out = net(data)
                optimizer.zero_grad()
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
                j += 1
                print("lr={} epoches= {},j={},loss is {}".format(lr,i + 1, j, loss))
                if j % 20 == 19:
                    n_loss = loss.item()
                    losses.append(n_loss)
                    x_label.append(j)
                    draw_pic(x_label, losses, )
        plt.savefig('train.png')
        train_time = time.time() - start_time
        return train_time

    def test(self, data_set):
        correct = 0
        total = 0
        for batch_i, data in enumerate(data_set):
            inputs, labels = data
            outputs = net(inputs)
            _,predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / total
        return accuracy

    def run_demo(self,input_data):
        output_data = net(input_data)
        result = torch.max(output_data, 1)
        return output_data, result



if __name__ == '__main__':
    batch_size = 32
    epoch = 1
    opt = 'Adam'
    learning_rate = 0.01
    train_loader, test_loader = load_data(batch_size)
    criterion = nn.CrossEntropyLoss()
    net = CNN()
    train_time = net.train(epoch_num=epoch, lr=learning_rate, opt=opt, criterion=criterion)

    torch.save(net, '/Users/jianqiaolong/Downloads/hr-mnist-pytorch/train_result/cnnmodel.pkl')  # 保存整个模型
    print("The train model is saved")
    net_state_dict = net.state_dict()  # 获取模型参数
    torch.save(net_state_dict, '/Users/jianqiaolong/Downloads/hr-mnist-pytorch/train_result/cnnmodeldict.pkl')  # 保存模型参数
    print("The paramenters of the model have been saved")

    train_accuracy = net.test(train_loader)
    test_accuracy = net.test(test_loader)
    detail = """
**************************************************************
            The training has been completed!
--------------------------------------------------------------   
            detail:
                epoch:{}       learning rate:{}
                batch:{}       optimizer:{}
--------------------------------------------------------------
            net struct:
{}
--------------------------------------------------------------
            train_accurancy:{}%
            test_accurancy:{}%   
            train_time:{}s
**************************************************************  
    """.format(epoch, learning_rate, batch_size, opt, net, train_accuracy*100, test_accuracy*100, train_time)
    print(detail)


