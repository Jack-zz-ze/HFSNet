# -*- coding:utf-8 -*-
# @Time       :2023/1/31 下午4:11
# @AUTHOR     :ZengZheng
# @FileName   :demo.py

import gc
import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
from scipy.io import loadmat
import scipy.io as scio
import numpy as np
import time
import os
from utils import train_patch, setup_seed, output_metric, print_args, train_epoch, valid_epoch
from model.HFSNet import HFSNet
import imageio
import warnings
import cv2
from read_data import readData_args
from config import parse_args
warnings.filterwarnings("ignore", category=UserWarning)


def train_1times():
    # -------------------------------------------------------------------------------
    Data1, Data2, TrLabel_10TIMES, TsLabel_10TIMES = readData_args(args)

    patchsize1 = args.patches1  # input spatial size for 2D-CNN
    pad_width1 = np.floor(patchsize1 / 2)
    pad_width1 = int(pad_width1)  # 8

    TrainPatch11, TrainPatch21, TrainPatchcanny, TrainLabel = train_patch(Data1, Data2, patchsize1, pad_width1, TrLabel_10TIMES)
    TestPatch11, TestPatch21, TestPatchcanny, TestLabel = train_patch(Data1, Data2, patchsize1, pad_width1, TsLabel_10TIMES)
    train_dataset = Data.TensorDataset(TrainPatch11, TrainPatch21, TrainPatchcanny, TrainLabel)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = Data.TensorDataset(TestPatch11, TestPatch21, TestPatchcanny, TestLabel)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    # data size
    print("height1={0},width1={1},band1={2}".format(height1, width1, band1))
    print("height2={0},width2={1},band2={2}".format(height2, width2, band2))
    # -------------------------------------------------------------------------------
    # create model

    def weights_init(m):
 
        if isinstance(m,(nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)

    model = HFSNet(HSI_channels=band1, LiDAR_channels=band2,
                                       num_classes=args.num_classes, patch_size=args.patches1, args=args)


    model.apply(weights_init)


    model = model.cuda()
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
    # -------------------------------------------------------------------------------
    # train and test
    if args.flag_test == 'train':
        BestAcc = 0
        val_acc = []
        print("start training")
        tic_all=0
        for epoch in range(args.epoches):
            # train model
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, train_loader, criterion, optimizer,args)
            OA1, AA1, Kappa1, CA1 = output_metric(tar_t, pre_t)
            print("Epoch: {:03d} | train_loss: {:.4f} | train_OA: {:.4f} | train_AA: {:.4f} | train_Kappa: {:.4f}"
                  .format(epoch + 1, train_obj, OA1, AA1, Kappa1))
            scheduler.step()


            if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
                model.eval()
                tar_v, pre_v = valid_epoch(model, test_loader, criterion)
                OA2, AA2, Kappa2, CA2 = output_metric(tar_v, pre_v)
                val_acc.append(OA2)
                print("Every 5 epochs' records:")
                print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA2, Kappa2))
                print(CA2)
                if OA2 > BestAcc:
                    torch.save(model.state_dict(), f'./weight/{args.dataset}_GLT_Net.pkl')
                    BestAcc = OA2

        model.eval()
        model.load_state_dict(torch.load(f'./weight/{args.dataset}_GLT_Net.pkl'))
        tar_v, pre_v = valid_epoch(model, test_loader, criterion)
        OA, AA, Kappa, CA = output_metric(tar_v, pre_v)

        all=[]
        all.append(OA)
        all.append(AA)
        all.append(Kappa)
        all.append(CA)

        print("Maxmial Accuracy: {:.4f} | index: {:.4f}".format(max(val_acc), val_acc.index(max(val_acc))))   
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa)) 



if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    setup_seed(args.seed)
    if not os.path.isdir('weight'):
        os.mkdir('weight')
    if args.training_mode == 'one_time':
        train_1times()




