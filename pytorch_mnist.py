 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import operator
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import math
import networkx as nx
import pandas as pd
import copy
import csv
import sys

from load_demo_img import get_input_img


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#def draw_pic(index, cost_data):
#    plt.ion()
#    plt.plot(index, cost_data,color='dodgerblue', marker='')
#    plt.title("cost figure")
#    plt.xlabel('item_time')
#    plt.ylabel('cost')
#    plt.show() 
#    plt.pause(0.000001)

def load_data(batch_size):
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    # 使用Compose对象组装多个变换：转为tensor，标准化
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)  # 指定训练集
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)  # 指定测试集
    # train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=data_transform, download=True)  # 指定训练集
    # test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=data_transform, download=True)  # 指定测试集
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    print(num_train)
    train_index, valid_index,test_index= indices[split:], indices[:split],indices[:split]
    print(len(train_index),len(valid_index),len(test_index))
# define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = train_sampler, num_workers = num_workers, drop_last=True)  # 训练集加载器
    valid_loader = DataLoader(train_dataset, batch_size = batch_size, sampler = valid_sampler, num_workers = num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers = num_workers, shuffle=False)  # 测试集加载器

    return train_loader, test_loader, valid_loader
    
#---------------------------------mlp model and train----------------------------------------------------
class Mlp(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(input_size, node1)
        self.fc2 = nn.Linear(node1, node2)
        self.fc3 = nn.Linear(node2, output_size)

    def forward(self, x):
        x = x.view(-1,input_size)#mnist/fashionmnist:-1,28*28 cifar10:32*32 # -1表示自动匹配
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)#dim=1表示对第一个维度求概率：对于一个size为（64,10）的数组，其第一个维度表示10
    

    
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

    def train(self, epoch_num, lr, criterion, opt):
        #x_label = []
        #losses = []
        #valid_losses = []
        #j = 0
        start_time = time.time()
        optimizer = self.check_opt(opt, lr)
        weight_dic={}
        gradient_dic={}
        train_history_dict_epoch={}
        train_history_dict={}
        for i in range(epoch_num):
            for data, label in train_loader:
                data = torch.autograd.Variable(data)
                label = torch.autograd.Variable(label)
                #label = label.reshape(-1,1) # 将一维数据变为二维数据（64）->(64,1)
                #one_hot= torch.zeros(data.shape[0],10).scatter(1,label,1)
                out = net(data)
                optimizer.zero_grad()
                #loss = criterion(out, label)
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
                #label = label.reshape(-1,1) # 将一维数据变为二维数据（64）->(64,1)
                #one_hot= torch.zeros(data.shape[0],10).scatter(1,label,1)
                out = net(data)
                optimizer.zero_grad()
                valid_loss = criterion(out, label)
                valid_loss.backward()
                optimizer.step()
                
            for data,label in test_loader:
                data = torch.autograd.Variable(data)
                label = torch.autograd.Variable(label)
                #label = label.reshape(-1,1) # 将一维数据变为二维数据（64）->(64,1)
                #one_hot= torch.zeros(data.shape[0],10).scatter(1,label,1)
                out = net(data)
                optimizer.zero_grad()
                test_loss = criterion(out, label)
                test_loss.backward()
                optimizer.step()
                
            train_accuracy_epoch = net.test(train_loader)
            valid_accuracy_epoch= net.test(valid_loader)
            test_accuracy_epoch = net.test(test_loader)
            train_history_dict_epoch={'train_acc': train_accuracy_epoch, 'train_loss': loss,
		                                'val_acc': valid_accuracy_epoch, 'valid_loss':valid_loss,
		                                'test_acc': test_accuracy_epoch, 'test_loss': test_loss
		                                }
            #update dict every epoch
            torch.set_printoptions(threshold=np.inf) 
            weight_dic.update({"model_epoch"+str(i+1): net.state_dict()})
            for name, parms in net.named_parameters():
                torch.set_printoptions(threshold=np.inf)
                gradient_dic.update({"model_epoch"+str(i+1)+" "+name:str(parms.grad)}) 
            train_history_dict.update({"model_epoch"+str(i+1): train_history_dict_epoch})
            
            #print weight/gradient/trainhistory
            with open(f1,"a+") as file: 
                for var_name in net.state_dict():
                        #print(var_name,'\t',net.state_dict()[var_name])
                    torch.set_printoptions(threshold=np.inf)
                    file.writelines('\n'+'epoch:'+str(i+1)+'\n'+var_name+'\n'+str(net.state_dict()[var_name]))
            with open(f2,"a+") as file: 
                for name, parms in net.named_parameters():
                    np.set_printoptions(threshold=sys.maxsize) 
                    torch.set_printoptions(threshold=np.inf)
                    file.writelines('\n'+'epoch:'+str(i+1)+'\n'+'-->name:'+name+'-->grad_requirs:'+str(parms.requires_grad)+' -->grad_value:'+str(parms.grad))
            with open(f3,"a+") as file: 
                file.writelines('\n'+'epoch:'+str(i+1)+'\n'+'train_acc:'+str(train_accuracy_epoch)+'\n'+'train_loss:'+str(loss)+'\n'+'valid_acc:'
                            +str(valid_accuracy_epoch)+'\n'+'valid_loss:'+str(valid_loss)+'\n'+'test_acc:'
                            +str(test_accuracy_epoch)+'\n'+'test_loss:'+str(test_loss))
                
        #save whole model parements       
        torch.save({'weight_dict':weight_dic,'gradient_dict':gradient_dic,'train_history_dict':train_history_dict},__PATH__) 
        #save parements seperately
        torch.save(weight_dic,'/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/train_result10/weight_dic.pkl')
        torch.save(gradient_dic,'/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/train_result10/gradient_dic.pkl')
        torch.save(train_history_dict,'/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/train_result10/train_history_dic.pkl')
        plt.savefig('train.png')
        train_time = time.time() - start_time
        return train_time

#---------------------------------caculate accuracy----------------------------------------------------
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
    

#---------------------------------verify code----------------------------------------------------
def predict_test(): 
    # Load the model that we saved at the end of the training loop 
    model = Mlp() 
    path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/train_result10/cnnmodel.pkl" 
    model.load_state_dict(torch.load(path)) 
     
    running_accuracy = 0 
    total = 0 
 
    with torch.no_grad(): 
        for data in test_loader: 
            inputs, outputs = data 
            outputs = outputs.to(torch.float32) 
            predicted_outputs = model(inputs) 
            _, predicted = torch.max(predicted_outputs, 1) 
            total += outputs.size(0) 
            running_accuracy += (predicted == outputs).sum().item() 
 
        print('Accuracy of the model based on the test set of',6000,'inputs is: %.2f %%' % (100 * running_accuracy / total))  
        
#---------------------------------initialize weights----------------------------------------------------
def _initialize_weights(self):
    for m in self.modules():
        #如果m是nn.Conv2d
        if isinstance(m, nn.Conv2d):
            #卷积核尺度与输出通道相乘
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #权重的初始化
            m.weight.data.normal_(0, math.sqrt(2. / n))
            #偏差的初始化，填充0
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(0.5)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
#---------------------------------Property of the neural nodes----------------------------------------------------
#If the disparity of output edges, the order of layers should be reversed e.g.(set layer1 in layer2 position, set layer2 in layer 1 position)
def nodesproperty(dictionary,layer1,layer2,flag):
    z_sum=0
    z1=[]
    if flag==1:
        s=dictionary.sum(axis=1)#For input egdes, find the sum of the rows
        for i in range (0,layer2):
            for j in range (0,layer1):
                z=(abs(dictionary[i][j])/abs(s[i]))**2 
                z_sum+=z  
            z1.append('{:.4f}'.format(z_sum))
            z_sum=0
    else:
        s=dictionary.sum(axis=0)#For output edges, sum the columns
        for i in range (0,layer2):
            for j in range (0,layer1):
                z=(abs(dictionary[j][i])/abs(s[i]))**2 
                z_sum+=z   
            z1.append('{:.4f}'.format(z_sum))
            z_sum=0
    return z1    
#---------------------------------Filter tensors based on mask----------------------------------------------------
#Filter tensors based on mask. 根据mask对张量进行筛选
#z is the matrix want to be filter
def filter_mask(z):
    a=torch.tensor(z)
    # get the mask
    less_one = (a < 1)
    larger_one = (a > 1)
    # output: select by mask
    print(less_one, torch.masked_select(a, less_one)) # <1 tensor
    print(larger_one, torch.masked_select(a, larger_one))  # >1 tensor    
 
#---------------------------------save csv file----------------------------------------------------   
def csvfilesave(csvfile_path,list_name):
    csvFile = open(csvfile_path, "a+",)
    try:
        writer = csv.writer(csvFile)
        writer.writerow(list_name)
    finally:
        csvFile.close()   
#---------------------------------main function---------------------------------------------------- 
if __name__ == '__main__':
    ##-------------------------train parameters setup-------------------------
    batch_size = 100
    epoch = 20         
    opt = 'Adam'
    learning_rate = 0.001
    valid_size = 0.1 
    num_workers=0
    input_size= 28*28
    output_size=10
    node1=5
    node2=5
    #-------------------------------------------------------------------------
    __PATH__='/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/train_result10/wholetrainhistory.pkl'
    train_loader, test_loader, valid_loader = load_data(batch_size)
    criterion = nn.CrossEntropyLoss()#nn.CrossEntropyLoss() 
    net =Mlp()
    net.apply(_initialize_weights)
    f1="/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/train_result10/weights.txt"
    f2="/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/train_result10/gradients.txt"
    f3="/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/train_result10/trainhistory.txt"
    with open(f1, 'r+') as file:
        file.truncate(0)
    with open(f2, 'r+') as file:
        file.truncate(0)
    with open(f3, 'r+') as file:
        file.truncate(0)
    train_time = net.train(epoch_num=epoch, lr=learning_rate, opt=opt, criterion=criterion)
    net_state_dict = net.state_dict()

    for name, param in net.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                torch.set_printoptions(threshold=np.inf)
                net_named_paremeters=("{}, gradient: {}".format(name, param.grad))
            else:
                print("{} has not gradient".format(name))
    #net_named_paremeters=net.named_parameters()
    torch.save(net_state_dict, '/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/train_result10/cnnmodel.pkl')  # 保存整个模型
    #torch.save(net_named_paremeters,'/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/train_result10/parameters.pkl')
    print("The train model is saved")
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
    a=[test_accuracy]        
    print(a)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/accuracy.csv',a)
    #---------------------------------varify code----------------------------------------------------
    predict_test()
    #---------------------------------Property of the neural nodes----------------------------------------------------
    dod=net_state_dict
    
    fc1_weight=dod["fc1.weight"]
    fc2_weight=dod["fc2.weight"]
    fc3_weight=dod["fc3.weight"]
    copy1_fc1_weight=copy.deepcopy(fc1_weight)
    copy2_fc1_weight=copy.deepcopy(fc1_weight)
    copy3_fc1_weight=copy.deepcopy(fc1_weight)
    copy4_fc1_weight=copy.deepcopy(fc1_weight)
    
    copy1_fc2_weight=copy.deepcopy(fc2_weight)
    copy2_fc2_weight=copy.deepcopy(fc2_weight)
    copy3_fc2_weight=copy.deepcopy(fc2_weight)
    copy4_fc2_weight=copy.deepcopy(fc2_weight)
    
    copy1_fc3_weight=copy.deepcopy(fc3_weight)
    copy2_fc3_weight=copy.deepcopy(fc3_weight)
    copy3_fc3_weight=copy.deepcopy(fc3_weight)
    copy4_fc3_weight=copy.deepcopy(fc3_weight)
    
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix 显示所有省略号的玩意
 
    #z1:Property of the neural nodes-- input edges
    #z2:hidden layer1-hidden layer2
    #z3:hidden layer2-output layer
    z1=nodesproperty(fc1_weight,input_size,node1,1)
    z2=nodesproperty(fc2_weight,node1,node2,1)
    z3=nodesproperty(fc3_weight,node2,output_size,1)
    #print('z1',z1,'\n','z2',z2,'\n','z3',z3)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001z1.csv',z1)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001z2.csv',z2)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001z3.csv',z3)
    # writerCSV=pd.DataFrame(columns=name_attribute,data=data)
    # writerCSV.to_csv('./no_fre.csv',encoding='utf-8')



    #---------------------------------Node Disparity of input edges----------------------------------------------------
    #positive disparity
    fc1_weight_positive=copy1_fc1_weight
    fc1_weight_positive[fc1_weight_positive<0]=0
    fc2_weight_positive=copy1_fc2_weight
    fc2_weight_positive[fc2_weight_positive<0]=0
    fc3_weight_positive=copy1_fc3_weight
    fc3_weight_positive[fc3_weight_positive<0]=0

    #z4:Property of the neural nodes-- input edges
    #z5:hidden layer1-hidden layer2
    #z6:hidden layer2-output layer
    z4=nodesproperty(fc1_weight_positive,input_size,node1,1)
    z5=nodesproperty(fc2_weight_positive,node1,node2,1)
    z6=nodesproperty(fc3_weight_positive,node2,output_size,1)
    #print('z4',z4,'\n','z5',z5,'\n','z6',z6)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001z4.csv',z4)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001z5.csv',z5)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001z6.csv',z6)            
    #negative disparity            
    fc1_weight_negative=copy2_fc1_weight
    fc1_weight_negative[fc1_weight_negative>0]=0
    fc2_weight_negative=copy2_fc2_weight
    fc2_weight_negative[fc2_weight_negative>0]=0
    fc3_weight_negative=copy2_fc3_weight
    fc3_weight_negative[fc3_weight_negative>0]=0
    
    #z7:Property of the neural nodes-- input edges
    #z8:hidden layer1-hidden layer2
    #z9:hidden layer2-output layer
    z7=nodesproperty(fc1_weight_negative,input_size,node1,1)
    z8=nodesproperty(fc2_weight_negative,node1,node2,1)
    z9=nodesproperty(fc3_weight_negative,node2,output_size,1)
    #print('z7',z7,'\n','z8',z8,'\n','z9',z9)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001z7.csv',z7)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001z8.csv',z8)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001z9.csv',z9)
    
   #---------------------------------Property of the neural nodes----------------------------------------------------
    o1=nodesproperty(fc1_weight,node1,input_size,0)
    o2=nodesproperty(fc2_weight,node2,node1,0)
    o3=nodesproperty(fc3_weight,output_size,node2,0)
    #print('o1',o1,'\n','o2',o2,'\n','o3',o3)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001o1.csv',o1)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001o2.csv',o2)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001o3.csv',o3)
    #---------------------------------Node Disparity of output edges----------------------------------------------------
    #positive disparity
    fc1_weight_positive_output=copy3_fc1_weight
    fc1_weight_positive_output[fc1_weight_positive_output<0]=0
    fc2_weight_positive_output=copy3_fc2_weight
    fc2_weight_positive_output[fc2_weight_positive_output<0]=0
    fc3_weight_positive_output=copy3_fc3_weight
    fc3_weight_positive_output[fc3_weight_positive_output<0]=0

    o4=nodesproperty(fc1_weight_positive_output,node1,input_size,0)
    o5=nodesproperty(fc2_weight_positive_output,node2,node1,0)
    o6=nodesproperty(fc3_weight_positive_output,output_size,node2,0)
    #print('o4',o4,'\n','o5',o5,'\n','o6',o6)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001o4.csv',o4)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001o5.csv',o5)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001o6.csv',o6)            
    #negative disparity            
    fc1_weight_negative_output=copy4_fc1_weight
    fc1_weight_negative_output[fc1_weight_negative_output>0]=0
    fc2_weight_negative_output=copy4_fc2_weight
    fc2_weight_negative_output[fc2_weight_negative_output>0]=0
    fc3_weight_negative_output=copy4_fc3_weight
    fc3_weight_negative_output[fc3_weight_negative_output>0]=0
      
    o7=nodesproperty(fc1_weight_negative_output,node1,input_size,0)
    o8=nodesproperty(fc2_weight_negative_output,node2,node1,0)
    o9=nodesproperty(fc3_weight_negative_output,output_size,node2,0)
    #print('o7',o7,'\n','o8',o8,'\n','o9',o9)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001o7.csv',o7)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001o8.csv',o8)
    csvfilesave('/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-test/b100l5lr0001o9.csv',o9)   
    #filter z1 matrix
    #filter_mask(z1)  
