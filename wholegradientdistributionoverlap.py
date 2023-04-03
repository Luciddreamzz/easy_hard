import numpy as np 
import matplotlib.mlab as mlab 
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
from scipy.stats import norm

import seaborn as sns 
import matplotlib as mpl 
from torch import tensor
torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix 显示所有省略号的玩意
#---------------------------------把字符串转换成张量 let string transform to tensor----------------------------------------------------
def stringtotensor(path,name):
    #fig, ax = plt.subplots()
    f=open(path,'rb')
    dod=torch.load(f)
    torch.set_printoptions(threshold=np.inf) #Display all data for complex matrix 显示所有省略号的玩意
    gradient=dod[name]
    start_index = gradient.find("[[")
    end_index = gradient.rfind("]]") + 2
    desired_string = gradient[start_index:end_index]
    # define the string representation of the array
    # create a numpy array from the string representation
    array = np.array(eval(desired_string))
# convert the numpy array to a tensor
    tensor = torch.from_numpy(array)
    return tensor

#---------------------------------Scott_rule规则用于确定直方图中的条柱宽度 The Scott_rule is used to determine the width of the bars in the histogram----------------------------------------------------
def Scott_rule(data):
    bin_width = 3.5 * np.std(data.numpy()) / len(data) ** (1/3)
    print(bin_width)
    return bin_width
#---------------------------------把所有张量写成一个 stack all tensor into one tensor----------------------------------------------------
def getbins(fc_weight):
    bins=np.arange(min(fc_weight).item(), max(fc_weight).item() + Scott_rule(fc_weight), Scott_rule(fc_weight))
    return bins

#---------------------------------把所有张量写成一个 add all tensor into one tensor----------------------------------------------------
def tensoradd(tensor1,tensor2,tensor3,tensor4,tensor5,tensor6,tensor7,tensor8,tensor9,tensor10):
# concatenate the two tensors along dimension 0
    concatenated_tensor = torch.cat((tensor1,tensor2,tensor3,tensor4,tensor5,tensor6,tensor7,tensor8,tensor9,tensor10), dim=0).flatten()
    print(len(concatenated_tensor))
    return concatenated_tensor
#---------------------------------画图----------------------------------------------------
def painthistoverlap(fc_weight_max,fc_weight_mid,fc_weight_min,figure_name,flag):
    
    plt.subplot(3,1,flag)

    #fig, ax = plt.subplots()
   
        #sns.set_palette("hls") 
        #mpl.rc("figure", figsize=(6,4)) 
    sns.distplot(fc_weight_max,bins=getbins(fc_weight_max),kde_kws={"color":"blueviolet", "lw":1 ,"linestyle":"-","label":"max accuracy"}, hist_kws={ "color": "blueviolet" }) #lightcoral mediumturquoise
    sns.distplot(fc_weight_mid,bins=getbins(fc_weight_mid),kde_kws={"color":"deepskyblue", "lw":1 ,"linestyle":"-","label":"mid accuracy"}, hist_kws={ "color": "deepskyblue" })
    sns.distplot(fc_weight_min,bins=getbins(fc_weight_min),kde_kws={"color":"lightcoral", "lw":1 ,"linestyle":"-","label":"min accuracy"}, hist_kws={ "color": "lightcoral" })  
    plt.xticks(fontproperties = 'Times New Roman', size = 6)
    plt.yticks(fontproperties = 'Times New Roman', size = 6)
    plt.title(figure_name,fontsize=8)
    plt.xlabel("gradients",fontsize=6)
    plt.ylabel("Frequency",fontsize=6)
    plt.legend(loc='best',fontsize=8)


#---------------------------------main function---------------------------------------------------- 

plt.figure(figsize=(9,9),dpi=300)
#---------------------------------max fc1 weight----------------------------------------------------
# create your two tensors
max_fc1_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result5/gradient_dic.pkl","model_epoch20 fc1.weight")
max_fc1_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result6/gradient_dic.pkl","model_epoch20 fc1.weight")
max_fc1_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result5/gradient_dic.pkl","model_epoch20 fc1.weight")
max_fc1_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result5/gradient_dic.pkl","model_epoch20 fc1.weight")
max_fc1_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result6/gradient_dic.pkl","model_epoch20 fc1.weight")
max_fc1_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result8/gradient_dic.pkl","model_epoch20 fc1.weight")
max_fc1_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result10/gradient_dic.pkl","model_epoch20 fc1.weight")
max_fc1_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result2/gradient_dic.pkl","model_epoch20 fc1.weight")
max_fc1_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result2/gradient_dic.pkl","model_epoch20 fc1.weight")
max_fc1_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result10/gradient_dic.pkl","model_epoch20 fc1.weight")
max_fc1_weight_sum=tensoradd(max_fc1_weight_tensor1,max_fc1_weight_tensor2,max_fc1_weight_tensor3,max_fc1_weight_tensor4,max_fc1_weight_tensor5,
          max_fc1_weight_tensor6,max_fc1_weight_tensor7,max_fc1_weight_tensor8,max_fc1_weight_tensor9,max_fc1_weight_tensor10)

#---------------------------------mid fc1 weight----------------------------------------------------
mid_fc1_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result10/gradient_dic.pkl","model_epoch20 fc1.weight")
mid_fc1_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result9/gradient_dic.pkl","model_epoch20 fc1.weight")
mid_fc1_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result4/gradient_dic.pkl","model_epoch20 fc1.weight")
mid_fc1_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result5/gradient_dic.pkl","model_epoch20 fc1.weight")
mid_fc1_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result8/gradient_dic.pkl","model_epoch20 fc1.weight")
mid_fc1_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result10/gradient_dic.pkl","model_epoch20 fc1.weight")
mid_fc1_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result8/gradient_dic.pkl","model_epoch20 fc1.weight")
mid_fc1_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result9/gradient_dic.pkl","model_epoch20 fc1.weight")
mid_fc1_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result1/gradient_dic.pkl","model_epoch20 fc1.weight")
mid_fc1_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result10/gradient_dic.pkl","model_epoch20 fc1.weight")
mid_fc1_weight_sum=tensoradd(mid_fc1_weight_tensor1,mid_fc1_weight_tensor2,mid_fc1_weight_tensor3,mid_fc1_weight_tensor4,mid_fc1_weight_tensor5,
          mid_fc1_weight_tensor6,mid_fc1_weight_tensor7,mid_fc1_weight_tensor8,mid_fc1_weight_tensor9,mid_fc1_weight_tensor10)

#---------------------------------min fc1 weight----------------------------------------------------
min_fc1_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result3/gradient_dic.pkl","model_epoch20 fc1.weight")
min_fc1_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result3/gradient_dic.pkl","model_epoch20 fc1.weight")
min_fc1_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result8/gradient_dic.pkl","model_epoch20 fc1.weight")
min_fc1_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result9/gradient_dic.pkl","model_epoch20 fc1.weight")
min_fc1_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result10/gradient_dic.pkl","model_epoch20 fc1.weight")
min_fc1_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result1/gradient_dic.pkl","model_epoch20 fc1.weight")
min_fc1_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result6/gradient_dic.pkl","model_epoch20 fc1.weight")
min_fc1_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result6/gradient_dic.pkl","model_epoch20 fc1.weight")
min_fc1_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result2/gradient_dic.pkl","model_epoch20 fc1.weight")
min_fc1_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result3/gradient_dic.pkl","model_epoch20 fc1.weight")
min_fc1_weight_sum=tensoradd(min_fc1_weight_tensor1,min_fc1_weight_tensor2,min_fc1_weight_tensor3,min_fc1_weight_tensor4,min_fc1_weight_tensor5,
          min_fc1_weight_tensor6,min_fc1_weight_tensor7,min_fc1_weight_tensor8,min_fc1_weight_tensor9,min_fc1_weight_tensor10)

#---------------------------------max fc2 weight----------------------------------------------------
# create your two tensors
max_fc2_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result5/gradient_dic.pkl","model_epoch20 fc2.weight")
max_fc2_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result6/gradient_dic.pkl","model_epoch20 fc2.weight")
max_fc2_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result5/gradient_dic.pkl","model_epoch20 fc2.weight")
max_fc2_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result5/gradient_dic.pkl","model_epoch20 fc2.weight")
max_fc2_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result6/gradient_dic.pkl","model_epoch20 fc2.weight")
max_fc2_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result8/gradient_dic.pkl","model_epoch20 fc2.weight")
max_fc2_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result10/gradient_dic.pkl","model_epoch20 fc2.weight")
max_fc2_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result2/gradient_dic.pkl","model_epoch20 fc2.weight")
max_fc2_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result2/gradient_dic.pkl","model_epoch20 fc2.weight")
max_fc2_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result10/gradient_dic.pkl","model_epoch20 fc2.weight")
max_fc2_weight_sum=tensoradd(max_fc2_weight_tensor1,max_fc2_weight_tensor2,max_fc2_weight_tensor3,max_fc2_weight_tensor4,max_fc2_weight_tensor5,
          max_fc2_weight_tensor6,max_fc2_weight_tensor7,max_fc2_weight_tensor8,max_fc2_weight_tensor9,max_fc2_weight_tensor10)

#---------------------------------mid fc2 weight----------------------------------------------------
mid_fc2_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result10/gradient_dic.pkl","model_epoch20 fc2.weight")
mid_fc2_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result9/gradient_dic.pkl","model_epoch20 fc2.weight")
mid_fc2_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result4/gradient_dic.pkl","model_epoch20 fc2.weight")
mid_fc2_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result5/gradient_dic.pkl","model_epoch20 fc2.weight")
mid_fc2_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result8/gradient_dic.pkl","model_epoch20 fc2.weight")
mid_fc2_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result10/gradient_dic.pkl","model_epoch20 fc2.weight")
mid_fc2_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result8/gradient_dic.pkl","model_epoch20 fc2.weight")
mid_fc2_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result9/gradient_dic.pkl","model_epoch20 fc2.weight")
mid_fc2_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result1/gradient_dic.pkl","model_epoch20 fc2.weight")
mid_fc2_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result10/gradient_dic.pkl","model_epoch20 fc2.weight")
mid_fc2_weight_sum=tensoradd(mid_fc2_weight_tensor1,mid_fc2_weight_tensor2,mid_fc2_weight_tensor3,mid_fc2_weight_tensor4,mid_fc2_weight_tensor5,
          mid_fc2_weight_tensor6,mid_fc2_weight_tensor7,mid_fc2_weight_tensor8,mid_fc2_weight_tensor9,mid_fc2_weight_tensor10)

#---------------------------------min fc2 weight----------------------------------------------------
min_fc2_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result3/gradient_dic.pkl","model_epoch20 fc2.weight")
min_fc2_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result3/gradient_dic.pkl","model_epoch20 fc2.weight")
min_fc2_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result8/gradient_dic.pkl","model_epoch20 fc2.weight")
min_fc2_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result9/gradient_dic.pkl","model_epoch20 fc2.weight")
min_fc2_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result10/gradient_dic.pkl","model_epoch20 fc2.weight")
min_fc2_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result1/gradient_dic.pkl","model_epoch20 fc2.weight")
min_fc2_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result6/gradient_dic.pkl","model_epoch20 fc2.weight")
min_fc2_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result6/gradient_dic.pkl","model_epoch20 fc2.weight")
min_fc2_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result2/gradient_dic.pkl","model_epoch20 fc2.weight")
min_fc2_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result3/gradient_dic.pkl","model_epoch20 fc2.weight")
min_fc2_weight_sum=tensoradd(min_fc2_weight_tensor1,min_fc2_weight_tensor2,min_fc2_weight_tensor3,min_fc2_weight_tensor4,min_fc2_weight_tensor5,
          min_fc2_weight_tensor6,min_fc2_weight_tensor7,min_fc2_weight_tensor8,min_fc2_weight_tensor9,min_fc2_weight_tensor10)


#---------------------------------max fc3 weight----------------------------------------------------
# create your two tensors
max_fc3_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result5/gradient_dic.pkl","model_epoch20 fc3.weight")
max_fc3_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result6/gradient_dic.pkl","model_epoch20 fc3.weight")
max_fc3_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result5/gradient_dic.pkl","model_epoch20 fc3.weight")
max_fc3_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result5/gradient_dic.pkl","model_epoch20 fc3.weight")
max_fc3_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result6/gradient_dic.pkl","model_epoch20 fc3.weight")
max_fc3_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result8/gradient_dic.pkl","model_epoch20 fc3.weight")
max_fc3_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result10/gradient_dic.pkl","model_epoch20 fc3.weight")
max_fc3_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result2/gradient_dic.pkl","model_epoch20 fc3.weight")
max_fc3_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result2/gradient_dic.pkl","model_epoch20 fc3.weight")
max_fc3_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result10/gradient_dic.pkl","model_epoch20 fc3.weight")
max_fc3_weight_sum=tensoradd(max_fc3_weight_tensor1,max_fc3_weight_tensor2,max_fc3_weight_tensor3,max_fc3_weight_tensor4,max_fc3_weight_tensor5,
          max_fc3_weight_tensor6,max_fc3_weight_tensor7,max_fc3_weight_tensor8,max_fc3_weight_tensor9,max_fc3_weight_tensor10)

#---------------------------------mid fc3 weight----------------------------------------------------
mid_fc3_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-1/train_result10/gradient_dic.pkl","model_epoch20 fc3.weight")
mid_fc3_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result9/gradient_dic.pkl","model_epoch20 fc3.weight")
mid_fc3_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result4/gradient_dic.pkl","model_epoch20 fc3.weight")
mid_fc3_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result5/gradient_dic.pkl","model_epoch20 fc3.weight")
mid_fc3_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result8/gradient_dic.pkl","model_epoch20 fc3.weight")
mid_fc3_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result10/gradient_dic.pkl","model_epoch20 fc3.weight")
mid_fc3_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result8/gradient_dic.pkl","model_epoch20 fc3.weight")
mid_fc3_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result9/gradient_dic.pkl","model_epoch20 fc3.weight")
mid_fc3_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result1/gradient_dic.pkl","model_epoch20 fc3.weight")
mid_fc3_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-10/train_result10/gradient_dic.pkl","model_epoch20 fc3.weight")
mid_fc3_weight_sum=tensoradd(mid_fc3_weight_tensor1,mid_fc3_weight_tensor2,mid_fc3_weight_tensor3,mid_fc3_weight_tensor4,mid_fc3_weight_tensor5,
          mid_fc3_weight_tensor6,mid_fc3_weight_tensor7,mid_fc3_weight_tensor8,mid_fc3_weight_tensor9,mid_fc3_weight_tensor10)

#---------------------------------min fc3 weight----------------------------------------------------
min_fc3_weight_tensor1 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-2/train_result3/gradient_dic.pkl","model_epoch20 fc3.weight")
min_fc3_weight_tensor2 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-3/train_result3/gradient_dic.pkl","model_epoch20 fc3.weight")
min_fc3_weight_tensor3 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result8/gradient_dic.pkl","model_epoch20 fc3.weight")
min_fc3_weight_tensor4 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result9/gradient_dic.pkl","model_epoch20 fc3.weight")
min_fc3_weight_tensor5 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-4/train_result10/gradient_dic.pkl","model_epoch20 fc3.weight")
min_fc3_weight_tensor6 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-5/train_result1/gradient_dic.pkl","model_epoch20 fc3.weight")
min_fc3_weight_tensor7 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-6/train_result6/gradient_dic.pkl","model_epoch20 fc3.weight")
min_fc3_weight_tensor8 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-7/train_result6/gradient_dic.pkl","model_epoch20 fc3.weight")
min_fc3_weight_tensor9 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-8/train_result2/gradient_dic.pkl","model_epoch20 fc3.weight")
min_fc3_weight_tensor10 = stringtotensor("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/b100l5lr0001-9/train_result3/gradient_dic.pkl","model_epoch20 fc3.weight")
min_fc3_weight_sum=tensoradd(min_fc3_weight_tensor1,min_fc3_weight_tensor2,min_fc3_weight_tensor3,min_fc3_weight_tensor4,min_fc3_weight_tensor5,
          min_fc3_weight_tensor6,min_fc3_weight_tensor7,min_fc3_weight_tensor8,min_fc3_weight_tensor9,min_fc3_weight_tensor10)





painthistoverlap(max_fc1_weight_sum,mid_fc1_weight_sum,min_fc1_weight_sum,"All gradients comparisons of layer 1",1)
painthistoverlap(max_fc2_weight_sum,mid_fc2_weight_sum,min_fc2_weight_sum,"All gradients comparisons of layer 2",2)
painthistoverlap(max_fc3_weight_sum,mid_fc3_weight_sum,min_fc3_weight_sum,"All gradients comparisons of layer 3",3)
plt.tight_layout()
plt.savefig("/Users/jianqiaolong/Downloads/hr-mnist-pytorch/result/whole_gradients_distribution_overlap.png")
plt.show()