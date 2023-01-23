import cv2 as cv
import numpy as np
import torch
import torchvision
##
def get_input_img(img_path):
    img = cv.imread(img_path)  # 读取图片
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_r = cv.resize(gray, (28, 28))  # 改变图像尺寸，以符合MNIST输入尺寸要求
    cv.normalize(img_r, img_r, alpha=1, beta=0,norm_type=cv.NORM_MINMAX)  # 将图像像素值标准化到[0,1]区间,起到图像阈值处理作用
    # cv.imshow('1', img_r)
    # cv.waitKey(0)
    img_r = (0.5-img_r)/0.5  # 将图像矩阵标准化到mean=0.5,std_val=0.5,像素值分布在[-1, 1]
    img_r =np.array(img_r).astype(np.float32)
    img_r = np.expand_dims(img_r, 0)
    img_r = np.expand_dims(img_r, 0)
    return img_r
if __name__ == '__main__':
    img_path = './test_real_data/00.png'
    out_img = get_input_img(img_path)
    print(out_img.shape)
