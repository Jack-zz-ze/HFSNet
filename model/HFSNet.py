# -*- coding:utf-8 -*-
# @Time       :2023/1/31 下午4:11
# @AUTHOR     :ZengZheng
# @FileName   :HFSNet.py

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torch.utils.checkpoint as checkpoint
import numpy as np
from model.construct import RLFB
from model.DSM_extract_module import DSM_sobel
from model.Transformer import Transformer
from einops import rearrange, repeat
import segmentation_models_pytorch as smp

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

class hconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=None, p1=64, p2=None, g=64):
        super(hconv, self).__init__()
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=g, padding=kernel_size//3, stride=stride)
        # self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p1, stride=stride)
        self.gwc1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p1, stride=stride)
        self.gwc2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p2, stride=stride)

    def forward(self, x):
        return self.gwc(x) + self.gwc1(x) + self.gwc2(x)


class decoder_construct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_construct, self).__init__()

        self.connet = nn.Sequential(
            RLFB(in_channels=in_channels,
                out_channels=out_channels),
            nn.Sigmoid(),
        )
        # self.loss_fun2 = nn.MSELoss()
        
        # self.loss_fun2 =DiceLoss()
        
        # 可用Pytorch自带的函数替代
        self.loss_fun2 = smp.losses.DiceLoss(mode='binary')


    def forward(self, HSI, DSM):
        HSI1 = self.connet(HSI)
        # return self.loss_fun2(HSI1, self.ssigmodi(DSM))
        return self.loss_fun2(HSI1, 1-DSM)

class HFSNet(nn.Module):
    def __init__(self, HSI_channels, LiDAR_channels, num_classes, patch_size=64, args=None):
        super(HFSNet, self).__init__()
        self.HSI_channels = HSI_channels
        self.LiDAR_channels = LiDAR_channels
        self.patch_size = patch_size
        self.dataset = args.dataset

        if args.dataset == 'Trento':
       
            HSI_channels=HSI_channels+1
            rgb_in_channels = 16   #patch=9
            self.convhsi_reduce2 = nn.Sequential(
                hconv(HSI_channels, rgb_in_channels,
                      p1=1,
                      p2=2,
                      g=4,    
                      ),
                nn.BatchNorm2d(rgb_in_channels),
                nn.ReLU()
            )
            inner_channel = [16, 32, 48]


        depth = [2, 2, 6]
        heads = [4, 8, 12]
        self.block1 = Block(
                            block_id='first',
                            out_channels=inner_channel[0],
                            # rgb_in_channels=self.HSI_channels,
                            rgb_in_channels=rgb_in_channels,
                            t_in_channels=self.LiDAR_channels,
                            patch_size=self.patch_size,
                            depth=depth[0],
                            heads=heads[0],
                            )

        self.block2 = Block(
                            out_channels=inner_channel[1],
                            rgb_in_channels=inner_channel[0],
                            t_in_channels=inner_channel[0],
                            patch_size=self.patch_size,
                            depth=depth[1],
                            heads=heads[1],
                            # patch_size=int(self.patch_size//2)
                            )

        self.block3 = Block(
                            out_channels=inner_channel[2],
                            rgb_in_channels=inner_channel[1],
                            t_in_channels=inner_channel[1],
                            patch_size=self.patch_size,
                            depth=depth[2],
                            heads=heads[2],
                            # patch_size=int(self.patch_size//2)
                            )


        self.cnn_classifier = CNN_Classifier(inner_channel[2], num_classes)
        self.decoder_construct = decoder_construct(inner_channel[2],1)


        self.coefficient1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([0.5]))

        self.shared1to2 = nn.Sequential(
            nn.Conv2d(inner_channel[0], inner_channel[1], 1, 1),
            nn.BatchNorm2d(inner_channel[1]),
            nn.ReLU()
        )


    def classifier(self, x):
        x_cls2 = self.cnn_classifier(x)
        return x_cls2

    def forward(self, HSI, DSM, datacanny, gt):

        # HSI_=HSI.unsqueeze(1)
        # HSI_reduce = self.convhsi_reduce(HSI_)
        if self.dataset == 'Trento':

            HSI_reduce = torch.cat([HSI,torch.zeros(HSI.shape)[:,1,:,:].reshape(HSI.shape[0],1,HSI.shape[2],HSI.shape[3]).cuda()],dim=1)

  
        HSI_reduce = HSI_reduce.reshape(HSI_reduce.shape[0],-1,self.patch_size,self.patch_size)
        HSI_reduce_ = self.convhsi_reduce2(HSI_reduce)

        HSI1, DSM1, shared1, _ = self.block1(HSI_reduce_, DSM, None)
        HSI2, DSM2, shared2, _ = self.block2(HSI1, DSM1, shared1) 
        shared1_2 = self.coefficient1 * self.shared1to2(shared1) + self.coefficient2 * shared2
        HSI3, DSM3, shared3, HSI_transformer_cls = self.block3(HSI2, DSM2, shared1_2)
        x = self.classifier(shared3)
        loss = self.decoder_construct(HSI3, datacanny)
        
        return x, loss



class CNN_Classifier(nn.Module):
    def __init__(self, inchannel, Classes):
        super(CNN_Classifier, self).__init__()
        # 32
        self.outchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, self.outchannel, 1),
            nn.BatchNorm2d(self.outchannel),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.outchannel, Classes, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        x_out = x
        return x_out


class Block(nn.Module):
    def __init__(self,
                 block_id=None,
                 rgb_in_channels=144,
                 out_channels=144,
                 t_in_channels=21,
                 depth=5,
                 heads=8,
                 patch_size=5
                 ):
        super(Block, self).__init__()
        self.block_id = block_id
        self.patch_size = patch_size


        # spectral
        self.en_transformer = Transformer(self.patch_size ** 2, depth=depth, heads=heads, dim_head=6,
                                          mlp_head=rgb_in_channels, dropout=0.1)
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, rgb_in_channels + 1, self.patch_size ** 2))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.patch_size ** 2))

        self.HSI_reducedim_linear = nn.Sequential(
            nn.LayerNorm(rgb_in_channels),
            nn.Linear(rgb_in_channels, out_channels),

        )
        self.dropout = nn.Dropout(0.1)

        # self.max_drop1 = nn.MaxPool2d(2)
        # self.max_drop2 = nn.MaxPool2d(2)
        self.t_conv = nn.Sequential(
                                    DSM_sobel(in_channels=t_in_channels, out_channels=out_channels)
                                    )


        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=rgb_in_channels, out_channels=out_channels, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.to_latent = nn.Identity()
        self.fused_height = Fused_height(out_channels)
        self.SPLIT_UPDATE = SPLIT_UPDATE(out_channels)

    def img2seq(self, x):
        [b, c, h, w] = x.shape
        x = x.reshape((b, c, h * w))
        return x

    def seq2img(self, x):
        [b, c, d] = x.shape
        p = int(d ** .5)
        x = x.reshape((b, c, p, p))
        return x

    # spectral
    def transfomer_HSI(self, x_):

        x_ = self.img2seq(x_)
        # x_ = torch.einsum('nld->ndl', x_)
        #
        # x_ = torch.einsum('nld->ndl', x_)
        b, n, _ = x_.shape
        # add pos embed w/o cls token
        x = x_ + self.encoder_pos_embed[:, 1:, :]

        # append cls token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # add position embedding
        x += self.encoder_pos_embed[:, :1]

        x = self.dropout(x)

        RGB_swin = self.en_transformer(x, mask=None)
        RGB_swin_cls = self.to_latent(RGB_swin[:, 0])
        RGB_swin_ = self.seq2img(RGB_swin[:, 1:, :])

        RGB_swin_=RGB_swin_.permute(0,2,3,1)
        RGB_swin_ = self.HSI_reducedim_linear(RGB_swin_)
        RGB_swin_ = RGB_swin_.permute(0, 3, 1, 2)
        return RGB_swin_, RGB_swin_cls


    def forward(self, HSI, DSM, shared):

        DSM_conv = self.t_conv(DSM)
        HSI_transformer,HSI_transformer_cls = self.transfomer_HSI(HSI)
        if self.block_id =='first':
            shared = torch.zeros(DSM_conv.shape).cuda()
        else:
            shared = self.shared_conv(shared)
        new_shared = self.fused_height(HSI_transformer, DSM_conv, shared)
        new_HSI, new_DSM, new_shared = self.SPLIT_UPDATE(HSI_transformer,new_shared,DSM_conv)
 
        return new_HSI, new_DSM, new_shared,HSI_transformer_cls


class channel_enhance(nn.Module):
    def __init__(self,kernel_size=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_hsi = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv_hsi(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        s_eca = x * y.expand_as(x)
        new_HSI = s_eca + x

        return new_HSI

class SPLIT_UPDATE(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channel_enhance = channel_enhance()
        self.rgb_distribute_1x1conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.t_distribute_1x1conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
    def forward(self, HSI, S, DSM):

        s_HSI = self.rgb_distribute_1x1conv(S - HSI)
        s_DSM = self.t_distribute_1x1conv(S - DSM)


        new_HSI = self.channel_enhance(s_HSI) + HSI
        new_DSM = self.channel_enhance(s_DSM) + DSM
        return new_HSI,new_DSM,S


class Fused_height(nn.Module):
    def __init__(self, channels):
        super(Fused_height, self).__init__()

        kernelsize = 3
        self.offsets = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=kernelsize * kernelsize * 2, kernel_size=kernelsize,
                                 stride=1, padding=1),
            # nn.BatchNorm2d(kernelsize * kernelsize * 2),
            # nn.ReLU(),
        )
        self.conv_mask = nn.Sequential(
            nn.Conv2d(channels, kernelsize * kernelsize, kernel_size=kernelsize, stride=1, padding=1),
            # nn.BatchNorm2d(kernelsize * kernelsize),
            # nn.ReLU(),
        )
        self.regular_conv = nn.Conv2d(channels, channels, kernel_size=kernelsize, stride=1, padding=1)

        self.conv_TT = nn.Conv2d(channels, channels, kernel_size=1)

        self.conv_BB = nn.Conv2d(channels, channels, kernel_size=1)


        self.conv_share = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.ReLU(inplace=True),

        )
       
        self.lamda1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.lamda2 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda
    def forward(self, HSI, DSM, shared):

        DSM = DSM * torch.sigmoid(self.lamda1) + HSI * torch.sigmoid(self.lamda2)

        offsets = self.offsets(DSM)
        mask = torch.sigmoid(self.conv_mask(DSM))
        DSM_guide_rgb_s = F.relu(torchvision.ops.deform_conv2d(input=HSI, offset=offsets,
                                                               weight=self.regular_conv.weight,
                                                               mask=mask, padding=1))



        DSM_guide_rgb_s_ = DSM_guide_rgb_s + self.conv_BB(DSM)
        new_shared = torch.cat([DSM_guide_rgb_s_, shared], dim=1)

        new_shared = self.conv_share(new_shared)
        return new_shared




if __name__ == "__main__":



    net = HFSNet(
        HSI_channels=144,
        LiDAR_channels=1,
        num_classes=11, patch_size=5
    )
    net=net.cuda()
    dummy_x = torch.randn(1, 144, 5, 5).cuda()
    dummy_y = torch.randn(1, 1, 5, 5).cuda()
    logits = net(dummy_x,dummy_y,dummy_y,dummy_y,dummy_y,dummy_y)  # (1,3)
    # print(net)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter:%.2fM" % (total / 1e6))
    print("Number of parameter:%.2f" % (total / 1e8))

    print(logits.shape)
