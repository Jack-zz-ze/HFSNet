# -*- coding:utf-8 -*-
# @Time       :2023/1/31 下午4:11
# @AUTHOR     :ZengZheng
# @FileName   :DSM_extract_module.py

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class Sobelxy(nn.Module):
    def __init__(self,channels,kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()


        sobel_filter = np.array([[1, 1, 1],
                                 [1, -8, 1],
                                 [1, 1, 1]])

        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding,
                             stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))


    def forward(self, x):
        sobelx = self.convx(x)

        x = torch.abs(sobelx)

        return x

class ConvBNGelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBNGelu2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                      stride=stride, dilation=dilation, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            # nn.ReLU(),  # No effect on order
            # nn.MaxPool2d(2),
        )

    def forward(self,x):
        # print(x.size())
        # return F.leaky_relu(self.conv(x), negative_slope=0.2)

        return self.conv(x)



class DSM_sobel(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(DSM_sobel, self).__init__()

        self.out_channels=out_channels
        # init mask
        self.dim = int(self.out_channels // 4)
        # self.sobel_conv = ConvBNGelu2d(in_channels, self.dim)
        self.sobel_conv = ConvBNGelu2d(in_channels, self.dim, kernel_size=1, padding=0)
        self.sobelconv = Sobelxy(self.dim)



        # self.sobel_conv2 = ConvBNGelu2d(self.dim, self.out_channels)
        self.sobel_conv2 = ConvBNGelu2d(self.dim, self.out_channels, kernel_size=1, padding=0)

      

        self.dense_conv1 = ConvBNGelu2d(in_channels, self.dim)
        self.dense_conv2 = ConvBNGelu2d(self.dim, self.dim)

        self.branch2_conv = ConvBNGelu2d(self.dim, self.out_channels)

        self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                      nn.BatchNorm2d(int(out_channels)),
                                      nn.GELU()
                                      )


    def forward(self, x):


        branch1 = self.sobelconv(self.sobel_conv(x))

        branch2 = self.dense_conv2(self.dense_conv1(x))
        branch2=self.branch2_conv(branch2*branch1)


        return self.sobel_conv2(branch1)+branch2 + self.shortcut(x)




    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


