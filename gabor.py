#-*-coding:utf-8-*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


#构建Gabor滤波器
def build_filters():
     filters = []
     ksize = [7,9,11,13,15,17] # gabor尺度，6个
     lamda = np.pi/2.0         # 波长
     for theta in np.arange(0, np.pi, np.pi / 4): #gabor方向，0°，45°，90°，135°，共四个
         for K in range(6):
             kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
             kern /= 1.5*kern.sum()
             filters.append(kern)
     plt.figure(1)

     #用于绘制滤波器
     for temp in range(len(filters)):
         plt.subplot(4, 6, temp + 1)
         plt.imshow(filters[temp])
     plt.show()
     return filters

#Gabor滤波过程
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC1,kern)
        np.maximum(accum, fimg, accum)
    return accum

#Gabor特征提取
def getGabor(img,filters):
    for i in range(len(filters)):
        res1 = process(img, filters[i])
        res.append(np.asarray(res1))

    #用于绘制滤波效果
    plt.figure(2)
    for temp in range(len(res)):
        plt.subplot(4,6,temp+1)
        plt.imshow(res[temp], cmap='gray' )
    plt.show()

    return res  #返回滤波结果,结果为24幅图，按照gabor角度排列

if __name__ == '__main__':
    filters = build_filters()
    img = cv2.imread('./test.png')
    getGabor(img, filters)
