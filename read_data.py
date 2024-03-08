# -*- coding:utf-8 -*-
# @Time       :2023/1/31 下午4:11
# @AUTHOR     :ZengZheng
# @FileName   :read_data.py
import scipy.io as scio
from scipy.io import loadmat
import imageio
import spectral as spy
import numpy as np
import torch
import cv2
from utils import sample_gt_percentage
def readData_args(args):
    # prepare data
    
    if args.dataset == 'Trento':
        DataPath1 = r'./dataset/Trento/HSI.mat'
        DataPath2 = r'./dataset/Trento/LiDAR.mat'
        Data1 = loadmat(DataPath1)['HSI']
        Data2 = loadmat(DataPath2)['LiDAR']


        gt = loadmat('./dataset/Trento/TRLabel.mat')['TRLabel']+loadmat('./dataset/Trento/TSLabel.mat')['TSLabel']
        TrLabel_10TIMES, TsLabel_10TIMES = sample_gt_percentage(gt, args)

        
        scio.savemat('./weight/Trento_HSI_train_gt_new.mat',{'train': TrLabel_10TIMES})
        scio.savemat('./weight/Trento_HSI_test_gt_new.mat', {'test': TsLabel_10TIMES})
        
        TrLabel_10TIMES = loadmat('./weight/Trento_HSI_train_gt_new.mat')['train']  # 349*1905
        TsLabel_10TIMES = loadmat('./weight/Trento_HSI_test_gt_new.mat')['test']  # 349*1905

    Data1 = Data1.astype(np.float32)
    Data2 = Data2.astype(np.float32)

    return Data1, Data2, TrLabel_10TIMES, TsLabel_10TIMES
