import torch
from main import CNN
import torch.nn.functional as F
from load_demo_img import get_input_img
import numpy as np
from load_trained_model import load_model



def recognize_input_image(img_path):
    model_path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/train_result/cnnmodel.pkl"
    # dict_path = "train_result\cnnmodeldict.pkl"
    # img_path = './test_real_data/m0.png'
    """
    net
    """
    net = torch.load(model_path)
    """
    input image
    """
    input_img = get_input_img(img_path)  # 加载图片
    input_img = torch.from_numpy(input_img)  # 将图片转为tensor
    input_img = torch.autograd.Variable(input_img)
    output_data = net(input_img)
    prob = F.softmax(output_data, dim=1)
    prob = torch.autograd.Variable(prob).numpy()
    print(prob)
    pred = np.argmax(prob)
    print(pred)
    return [prob,pred]

if __name__ == '__main__':
    model_path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/train_result/cnnmodel.pkl"
    dict_path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/train_result/cnnmodel.pkl"
    img_path = '/Users/jianqiaolong/Downloads/hr-mnist-pytorch/test_real_data/m8.png'
    """
    net
    """
    net = torch.load(model_path)
    """
    input image
    """
    input_img = get_input_img(img_path)  # 加载图片
    input_img = torch.from_numpy(input_img) # 将图片转为tensor
    input_img = torch.autograd.Variable(input_img)
    output_data = net(input_img)
    prob = F.softmax(output_data, dim=1)
    prob = torch.autograd.Variable(prob).numpy()
    print(prob)
    pred = np.argmax(prob)
    print(pred)
