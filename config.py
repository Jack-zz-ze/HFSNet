# -*- coding:utf-8 -*-
# @Time       :2023/1/31 下午4:11
# @AUTHOR     :ZengZheng
# @FileName   :config.py

import argparse

def parse_args():
    # -------------------------------------------------------------------------------
    # Parameter Setting
    parser = argparse.ArgumentParser("HFSNet")
    parser.add_argument('--gpu_id', default='0', help='gpu id')
    parser.add_argument('--seed', type=int, default=0, help='number of seed')
    parser.add_argument('--dataseed', type=int, default=0, help='number of seed')
    parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
    parser.add_argument('--epoches', type=int, default=300, help='epoch number')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')  # diffGrad 1e-3
    parser.add_argument('--num', type=int, default=0, help='number')  
    parser.add_argument('--train_size', type=int, default=20, help='number')
    parser.add_argument('--lamba', type=int, default=10, help='number')


    parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parser.add_argument('--percentage', type=float, default=0.1, help='percentage')

    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')

    parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston'], default='Trento', help='dataset to use')
    parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
    parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')

    parser.add_argument('--patches1', type=int, default=9, help='number1 of patches')
    # parser.add_argument('--patches1', type=int, default=7, help='number1 of patches')
 
    parser.add_argument('--mse', type=int, default=1, help='mse')
    parser.add_argument('--training_mode', choices=['one_time', 'ten_times', 'test_all', 'train_standard'], default='one_time', help='training times')
    return parser.parse_args()