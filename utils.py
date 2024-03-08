# -*- coding:utf-8 -*-
# @Time       :2023/1/31 下午4:11
# @AUTHOR     :ZengZheng
# @FileName   :utils.py
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
import imageio
import sklearn.model_selection
import os
import random
def sample_gt_percentage(gt,args):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt) #r c of value index
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    #
    # if train_size > 1:
    #    train_size = int(train_size)

    # print("Sampling with train size = {}".format(train_size))
    train_indices, test_indices = [], []
    for c in np.unique(gt):
        if c == 0:
            continue
        indices = np.nonzero(gt == c)
        X = list(zip(*indices))  # x,y features
        train_size = int(len(X)*args.percentage)
        if train_size<=0:
            train_size=1
        train_size = 20
        print("Sampling  category {} with train size = {}".format(c, train_size))
        train, test = sklearn.model_selection.train_test_split(X, train_size=train_size,
                                                               random_state=args.dataseed)

        train_indices += train
        test_indices += test

    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]
    train_gt[train_indices] = gt[train_indices]
    test_gt[test_indices] = gt[test_indices]

    return train_gt, test_gt


    
def normalize(input2):
    input2_normalize = np.zeros(input2.shape)
    for i in range(input2.shape[2]):
        input2_max = np.max(input2[:, :, i])
        input2_min = np.min(input2[:, :, i])
        input2_normalize[:, :, i] = (input2[:, :, i] - input2_min) / (input2_max - input2_min)

    return input2_normalize


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False

    # os.environ['PYTHONHASHSEED'] = str(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True



# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

 
    return mirror_hsi

def train_patch(Data1, Data2, patchsize, pad_width, Label):
    [m1, n1, l1] = np.shape(Data1)
    # spy.save_rgb('lidar2013.jpg', img_lidar)

    imageio.imwrite("./dataset/aa_lidar.png",Data2.astype('uint8'))
    img_lidar = cv2.imread("./dataset/aa_lidar.png")



    img_lidar_canny = cv2.Canny(img_lidar, threshold1=2.5, threshold2=5, apertureSize=7) / 255
    imageio.imwrite("./dataset/aa_lidar_canny.png", img_lidar_canny.astype('uint8'))
    img_lidar_canny = img_lidar_canny.reshape([m1, n1, -1])
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)
    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2



    x1_pad=mirror_hsi(m1, n1, l1, x1, patch=patchsize)
    x2_pad = mirror_hsi(m2, n2, l2, x2, patch=patchsize)
    img_lidar_canny_pad = mirror_hsi(m2, n2, l2, img_lidar_canny, patch=patchsize)

    # construct the training and testing set
    [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')
    TrainPatch_canny = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    odd = True
    for i in range(len(ind1)):
        if odd:
            patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width+1), (ind4[i] - pad_width):(ind4[i] + pad_width+1), :]
        else:
            patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        if odd:
            patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),(ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        else:
            patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]

        patch2 = np.transpose(patch2, (2, 0, 1))
        TrainPatch2[i, :, :, :] = patch2
        if odd:
            patch3 = img_lidar_canny_pad[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),(ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        else:
            patch3 = img_lidar_canny_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]

        patch3 = np.transpose(patch3, (2, 0, 1))
        TrainPatch_canny[i, :, :,:] = patch3

        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainPatch_canny = torch.from_numpy(TrainPatch_canny)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return TrainPatch1, TrainPatch2, TrainPatch_canny,TrainLabel


def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))


def train_epoch(model, train_loader, criterion, optimizer, args):
    objs = AverageMeter()
    top1 = AverageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data11, batch_data21, batch_datacanny, batch_target) in enumerate(train_loader):
        batch_data11 = batch_data11.cuda()
        batch_data21 = batch_data21.cuda()
        batch_datacanny = batch_datacanny.cuda()
        batch_target = batch_target.cuda()
        optimizer.zero_grad()
        batch_pred, con_loss = model(batch_data11, batch_data21, batch_datacanny, batch_target)

        if args.mse:
            loss = criterion(batch_pred, batch_target) + args.lamba/10*con_loss
        else:
            loss = criterion(batch_pred, batch_target)
        # loss = con_loss[0]
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data11.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


def valid_epoch(model, valid_loader, criterion):
    objs = AverageMeter()
    top1 = AverageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data11, batch_data21, batch_datacanny, batch_target) in enumerate(valid_loader):
        batch_data11 = batch_data11.cuda()
        batch_data21 = batch_data21.cuda()
        batch_datacanny = batch_datacanny.cuda()
        batch_target = batch_target.cuda()

        batch_pred, con_loss = model(batch_data11, batch_data21, batch_datacanny, batch_target)

        loss = criterion(batch_pred, batch_target)


        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data11.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre


