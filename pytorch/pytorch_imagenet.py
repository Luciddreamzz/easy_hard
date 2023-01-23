#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets




#def draw_pic(index, cost_data):
#    plt.ion()
#    plt.plot(index, cost_data,color='dodgerblue', marker='')
#    plt.title("cost figure")
#    plt.xlabel('item_time')
#    plt.ylabel('cost')
#    plt.show() 
#    plt.pause(0.000001)

def load_data(batch_size):
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    train_dataset = datasets.ImageNet(root='./data', transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,]), download=True)  # 指定训练集
    test_dataset = datasets.ImageNet(root='./data', transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,]), download=True)  # 指定训练集
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    print(num_train)
    train_index, valid_index = indices[split:], indices[:split]
    print(len(train_index),len(valid_index))
# define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = train_sampler, num_workers = num_workers, drop_last=True)  # 训练集加载器
    valid_loader = DataLoader(train_dataset, batch_size = batch_size, sampler = valid_sampler, num_workers = num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试集加载器

    return train_loader, test_loader, valid_loader

class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(512*1*1, 30)#mnist/fashionmnist:28*28 cifar10:32*32
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 10)

    def forward(self, x):
        x = x.view(-1,512*1*1)#mnist/fashionmnist:-1,28*28 cifar10:32*32
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

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

    def train(self, epoch_num, lr, criterion=nn.CrossEntropyLoss(), opt='SGD'):
        #x_label = []
        #losses = []
        #valid_losses = []
        #j = 0
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
                train_loss=loss
                #j += 1
                #print("lr={} epoches= {},j={},loss is {}".format(lr,i + 1, j, loss))
                #if j % 20 == 19:
                #    n_loss = loss.item()
                #    losses.append(n_loss)
                #    x_label.append(j)
                #    draw_pic(x_label, losses, )
            
            for data,label in valid_loader:
                data = torch.autograd.Variable(data)
                label = torch.autograd.Variable(label)
                out = net(data)
                optimizer.zero_grad()
                valid_loss = criterion(out, label)
                valid_loss.backward()
                optimizer.step()
                
            for data,label in test_loader:
                data = torch.autograd.Variable(data)
                label = torch.autograd.Variable(label)
                out = net(data)
                optimizer.zero_grad()
                test_loss = criterion(out, label)
                test_loss.backward()
                optimizer.step()
                
            train_accuracy_epoch = net.test(train_loader)
            valid_accuracy_epoch= net.test(valid_loader)
            test_accuracy_epoch = net.test(test_loader)
            with open(f1,"a+") as file: 
                for var_name in net.state_dict():
                        #print(var_name,'\t',net.state_dict()[var_name])
                    file.writelines('\n'+'epoch:'+str(i+1)+'\n'+var_name+'\n'+str(net.state_dict()[var_name]))
            with open(f2,"a+") as file: 
                for name, parms in net.named_parameters(): 
                        #print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                        #        ' -->grad_value:',parms.grad)
                    file.writelines('\n'+'epoch:'+str(i+1)+'\n'+'-->name:'+name+'-->grad_requirs:'+str(parms.requires_grad)+' -->grad_value:'+str(parms.grad))
            with open(f3,"a+") as file: 
                file.writelines('\n'+'epoch:'+str(i+1)+'\n'+'train_acc:'+str(train_accuracy_epoch)+'\n'+'train_loss:'+str(loss)+'\n'+'valid_acc:'
                            +str(valid_accuracy_epoch)+'\n'+'valid_loss:'+str(valid_loss)+'\n'+'test_acc:'
                            +str(test_accuracy_epoch)+'\n'+'test_loss:'+str(test_loss))

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
    batch_size = 50
    test_batch_size = 64
    input_size = 3072
    epoch = 2
    opt = 'Adam'
    learning_rate = 0.001
    valid_size = 0.1 
    num_workers=8
    train_loader, test_loader, valid_loader = load_data(batch_size)
    criterion = nn.CrossEntropyLoss()
    net =Mlp()
    f1="/Users/jianqiaolong/Downloads/hr-mnist-pytorch/train_result/weights.txt"
    f2="/Users/jianqiaolong/Downloads/hr-mnist-pytorch/train_result/gradients.txt"
    f3="/Users/jianqiaolong/Downloads/hr-mnist-pytorch/train_result/trainhistory.txt"
    with open(f1, 'r+') as file:
        file.truncate(0)
    with open(f2, 'r+') as file:
        file.truncate(0)
    with open(f3, 'r+') as file:
        file.truncate(0)
    train_time = net.train(epoch_num=epoch, lr=learning_rate, opt=opt, criterion=criterion)

    torch.save(net, '/Users/jianqiaolong/Downloads/hr-mnist-pytorch/train_result/cnnmodel.pkl')  # 保存整个模型
    print("The train model is saved")
    #net_state_dict = net.state_dict()  # 获取模型参数
    
    #for var_name in net.state_dict():
    #    print(var_name,'\t',net.state_dict()[var_name])
    #for name, parms in net.named_parameters(): 
    #    print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
    #                ' -->grad_value:',parms.grad)

    # torch.save(net_state_dict, '/Users/jianqiaolong/Downloads/hr-mnist-pytorch/train_result/cnnmodeldict.pkl')  # 保存模型参数
    print("The paramenters of the model have been saved")

    train_accuracy = net.test(train_loader)
    test_accuracy = net.test(test_loader)
    valid_accuracy= net.test(valid_loader)
    print("Training the model...")
    
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
            valid_accurancy:{}% 
            train_time:{}s
**************************************************************  
    """.format(epoch, learning_rate, batch_size, opt, net, train_accuracy*100, test_accuracy*100, valid_accuracy*100, train_time)
    print(detail)
