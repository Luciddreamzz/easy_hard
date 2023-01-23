import torch
from main import CNN

def load_model(dict_path):
    # net = torch.load(model_path)
    state_dict = torch.load(dict_path)  # 加载模型参数
    net = CNN()  # 生成实例
    net.load_state_dict(state_dict) # 将参数载入模型
    return net



if __name__ == '__main__':
    model_path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/train_result/cnnmodel.pkl"
    dict_path = "/Users/jianqiaolong/Downloads/hr-mnist-pytorch/train_result/cnnmodeldict.pkl"
    load_model(model_path, dict_path)