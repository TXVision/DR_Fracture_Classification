##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: TXVision
## Email: cweidao@infervion.com
## Copyright (c) 2021
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import sys

import os
import time
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import logging
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from pywt import dwt2, idwt2

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.autograd import Variable

import encoding
from encoding.nn import LabelSmoothing, NLLMultiLabelSmooth
from encoding.utils import (accuracy, AverageMeter, MixUpWrapper, LR_Scheduler, torch_dist_sum)
from skimage.transform import resize
import random
from copy import deepcopy

import imgaug as ia
import imgaug.augmenters as iaa

import torchvision.models


# 随机镜像
def augment_mirroring(sample_data, axes=(0, 1, 2)):
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:] = sample_data[::-1]

    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]

    if 2 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]

    return sample_data

# 对比度增强
def augment_contrast(data_sample, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
    if not per_channel:
        mn = data_sample.mean()
        if preserve_range:
            minm = data_sample.min()
            maxm = data_sample.max()
        if np.random.random() < 0.5 and contrast_range[0] < 1:
            factor = np.random.uniform(contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
        data_sample = (data_sample - mn) * factor + mn
        if preserve_range:
            data_sample[data_sample < minm] = minm
            data_sample[data_sample > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            mn = data_sample[c].mean()
            if preserve_range:
                minm = data_sample[c].min()
                maxm = data_sample[c].max()
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            data_sample[c] = (data_sample[c] - mn) * factor + mn
            if preserve_range:
                data_sample[c][data_sample[c] < minm] = minm
                data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample

# brightness
def augment_brightness_multiplicative(data_sample, multiplier_range=(0.5, 2), per_channel=True):
    multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
    if not per_channel:
        data_sample *= multiplier
    else:
        for c in range(data_sample.shape[0]):
            multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
            data_sample[c] *= multiplier
    return data_sample

# rician_noise
def augment_rician_noise(data_sample, noise_variance=(0, 0.1)):
    variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = np.sqrt(
        (data_sample + np.random.normal(0.0, variance, size=data_sample.shape)) ** 2 +
        np.random.normal(0.0, variance, size=data_sample.shape) ** 2)
    return data_sample

# gaussian_noise
def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='DR_Fracture_Classification')
        parser.add_argument('--dataset', type=str, default='imagenet',
                            help='training dataset (default: imagenet)')
        parser.add_argument('--num_classes', type=int, metavar='N',
                            help='num_classes')
        parser.add_argument('--fold', type=int, metavar='N',
                            help='the fold of K-Fold')
        parser.add_argument('--has_aug', type=int, default=0, metavar='aug',
                            help='if perform augmentation')

        parser.add_argument('--csv-path', type=str,
                            help='the csv file contained all clinical info of pids')
        parser.add_argument('--base-size', type=int, default=None,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=224,
                            help='crop image size')
        parser.add_argument('--label-smoothing', type=float, default=0.0,
                            help='label-smoothing (default eta: 0.0)')
        parser.add_argument('--mixup', type=float, default=0.0,
                            help='mixup (default eta: 0.0)')
        parser.add_argument('--rand-aug', action='store_true',
                            default=False, help='random augment')
        # model params
        parser.add_argument('--model', type=str, default='densenet',
                            help='network model type (default: densenet)')
        parser.add_argument('--rectify', action='store_true',
                            default=False, help='rectify convolution')
        parser.add_argument('--rectify-avg', action='store_true',
                            default=False, help='rectify convolution')
        parser.add_argument('--pretrained', action='store_true',
                            default=False, help='load pretrianed mode')
        parser.add_argument('--last-gamma', action='store_true', default=False,
                            help='whether to init gamma of the last BN layer in \
                            each bottleneck to 0 (default: False)')
        parser.add_argument('--dropblock-prob', type=float, default=0,
                            help='DropBlock prob. default is 0.')
        parser.add_argument('--final-drop', type=float, default=0,
                            help='final dropout prob. default is 0.')
        # training params
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='batch size for training (default: 128)')
        parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                            help='batch size for testing (default: 256)')
        parser.add_argument('--epochs', type=int, default=120, metavar='N',
                            help='number of epochs to train (default: 600)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='the epoch number to start (default: 1)')
        parser.add_argument('--workers', type=int, default=8,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--log', dest='log_file', type=str, default=os.path.join(os.getcwd(), "train.log"),
                            help='save training log to file')
        # optimizer
        parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                            help='learning rate (default: 0.1)')
        parser.add_argument('--lr-scheduler', type=str, default='cos',
                            help='learning rate scheduler (default: cos)')
        parser.add_argument('--warmup-epochs', type=int, default=0,
                            help='number of warmup epochs (default: 0)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='SGD weight decay (default: 1e-4)')
        parser.add_argument('--no-bn-wd', action='store_true',
                            default=False, help='no bias decay')
        # seed
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        # distributed
        parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default=False,
                            help='evaluating')
        parser.add_argument('--export', type=str, default=None,
                            help='put the path to resuming file if needed')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args


def augment_rot90_multiModality(data_list, seg_list=[], num_rot=(1, 2, 3), axes=(0, 1, 2)):
    num_rot = np.random.choice(num_rot)
    axes = np.random.choice(axes, size=2, replace=False)
    axes.sort()
    new_data_list = []
    new_seg_list = []
    for i in range(len(data_list)):
        sample_data = np.rot90(data_list[i], num_rot, axes)
        # print('***', sample_data.shape)
        new_data_list.append(deepcopy(sample_data))
        if len(seg_list) > 0:
            sample_seg = np.rot90(seg_list[i], num_rot, axes)
            # print('++++', sample_seg.shape)
            new_seg_list.append(deepcopy(sample_seg))

    return new_data_list, new_seg_list


# 特定对z轴随机镜像
def augment_mirroring_z_multiModality(data_list, seg_list=[]):
    new_data_list = []
    new_seg_list = []

    if np.random.uniform() < 1:
        for i in range(len(data_list)):
            sample_data = data_list[i]
            sample_data[:] = sample_data[::-1]
            # print(sample_data.shape)
            new_data_list.append(sample_data)
            if len(seg_list) > 0:
                sample_seg = seg_list[i]
                sample_seg[:] = sample_seg[::-1]
                # print('seg', sample_seg.shape)
                new_seg_list.append(sample_seg)

    return new_data_list, new_seg_list


def get_2D_gaussian_map(im_h, im_w):
    from numpy import matlib as mb
    IMAGE_WIDTH = im_w
    IMAGE_HEIGHT = im_h

    center_x = IMAGE_WIDTH/2
    center_y = IMAGE_HEIGHT/2

    R = np.sqrt(center_x**2 + center_y**2)

    # Gauss_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    # 直接利用矩阵运算实现
    mask_x = mb.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)
    mask_y = mb.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)

    x1 = np.arange(IMAGE_WIDTH)
    x_map = mb.repmat(x1, IMAGE_HEIGHT, 1)

    y1 = np.arange(IMAGE_HEIGHT)
    y_map = mb.repmat(y1, IMAGE_WIDTH, 1)
    y_map = np.transpose(y_map)
    Gauss_map = np.sqrt((x_map-mask_x)**2+(y_map-mask_y)**2)
    Gauss_map = np.exp(-0.5*Gauss_map/R)
    return Gauss_map

def multiScaleSharpen_v1(img, radius=5):
    img = np.float32(img)
    Dest_float_img = np.zeros(img.shape, dtype=np.float32)+114

    w1 = 0.5
    w2 = 0.5
    w3 = 0.25

    GaussBlue1 = np.float32(cv2.GaussianBlur(img, (radius, radius), 1))
    GaussBlue2 = np.float32(cv2.GaussianBlur(img, (radius * 2 - 1, radius * 2 - 1), 2))
    GaussBlue3 = np.float32(cv2.GaussianBlur(img, (radius * 4 - 1, radius * 4 - 1), 4))

    D1 = img - GaussBlue1
    D2 = GaussBlue1 - GaussBlue2
    D3 = GaussBlue2 - GaussBlue3
    D1_mask = (D1 > 0) + (-1) * (D1 <= 0) + 0.0
    Dest_float_img = (1 - w1 * D1_mask) * D1 + w2 * D2 + w3 * D3 + img

    Dest_img = cv2.convertScaleAbs(Dest_float_img)
    return Dest_img

def edge_demo(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edge_output = cv2.Canny(blurred, 50, 150)
    return edge_output.copy()

class DRimgDataset(Dataset):
    def __init__(self, index_list, data_shape=256, label_name='label', mode='train', csv_path='',
                 reverse=False, has_aug=False):
        super(DRimgDataset, self).__init__()
        self.mode = mode
        self.data_shape=data_shape
        self.reverse = reverse
        self.label_name = label_name
        self.index_list = index_list
        self.add_gaussian_mask=False
        self.add_edge=False
        self.detail_enhancement=False
        self. wavelet_trans=False
        self.padding=True
        self.resize=True
        self.mosaic=False
        self.has_aug=has_aug
        self.random_rotate=True
        self.random_lightness=True
        self.random_transpose=True
        self.random_mirror=True
        self.random_brightness=False
        self.random_gaussian_noise=False
        self.random_rician_noise=False
        self.len = len(index_list)
        self.all_df = pd.read_csv(csv_path)

        print('=== mode:' + self.mode)
        print('=== num of samples: ', self.len)
        print('=== num of l1 samples: ', len([item for item in self.index_list if int(item.split('_')[-1].split('.')[0])==1]))
        print('=== num of l2 samples: ', len([item for item in self.index_list if int(item.split('_')[-1].split('.')[0])==2]))

    def load_mosaic(self, index):
        # loads images in a mosaic
        s = self.data_shape
        xc, yc = [int(random.uniform(s * 0.75, s * 1.25)) for _ in range(2)]  # mosaic center x, y
        ll_ = int(index.split('_')[-1].split('.')[0])
        s_list = [item for item in self.index_list if int(item.split('_')[-1].split('.')[0])==ll_]
        # s_list = list(self.all_df[(self.all_df['dataset'] == 'develop') & (self.all_df['label'] == ll_)]['pid'])

        np.random.shuffle(s_list)
        indices = [index] + s_list[:3] # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

        return img4


    def load_image_0(self, index):
        sample = cv2.imread(index)
        h0, w0 = sample.shape[:2]
        if self.detail_enhancement:
            sample = multiScaleSharpen_v1(sample, 5)

        if self.padding:
            height, width, num_channel = sample.shape
            max_edge = max(height, width)
            new_pix = np.zeros((max_edge, max_edge, num_channel), dtype=np.uint8)+114
            if self.add_edge:
                edge_ = edge_demo(sample)
                sample[:, :, 2] = edge_.astype(np.uint8) * 255


            if self.mode == 'train':
                if height > width:
                    random_bias_range = max_edge - width
                else:
                    random_bias_range = max_edge - height
                random_bias = np.random.randint(random_bias_range)
                if height > width:
                    new_pix[0:height, random_bias:random_bias + width, :] = sample[0:height, 0:width, :]
                else:
                    new_pix[random_bias:random_bias + height, 0:width, :] = sample[0:height, 0:width, :]
            else:
                new_pix[0:height, 0:width, :] = sample[0:height, 0:width, :]
        else:
            new_pix = sample
        if self.resize:
            new_pix = cv2.resize(new_pix, (self.data_shape, self.data_shape))
        return new_pix, (h0, w0), new_pix.shape[:2]


    def __getitem__(self, index):
        file_path = self.index_list[index]
        if self.mode=='train' and self.mosaic and np.random.random()>0.5:
        # if 0:
            # Load mosaic
            new_pix = self.load_mosaic(file_path)
            if self.resize:
                new_pix = cv2.resize(new_pix, (self.data_shape, self.data_shape))
        else:
            new_pix, _, _ = self.load_image_0(file_path)

        if self.mode=='train' and self.has_aug:
            if self.random_rotate:
                # 随机旋转
                num_rot=(0, 1, 2, 3)
                num_rot = np.random.choice(num_rot)
                if num_rot>0:
                    new_pix = np.rot90(new_pix, num_rot, axes=(0,1))
            if self.random_lightness:
                # random lightness augmentation
                randint = np.random.uniform(low=-10, high=10)
                randint = np.round(randint)
                new_pix = new_pix + randint
                # new_pix[:,:,:2] = new_pix[:,:,:2] + randint
            if self.random_transpose:
                if np.random.random() > 0.5:
                    new_pix = np.transpose(new_pix, (1, 0, 2))
            if self.random_mirror:
                new_pix = augment_mirroring(new_pix, axes=(0, 1))
            if self.random_brightness:
                new_pix=augment_brightness_multiplicative(new_pix, multiplier_range=(0.5, 2), per_channel=True)
            if self.random_gaussian_noise:
                new_pix=augment_gaussian_noise(new_pix, noise_variance=(0, 0.1))
            if self.random_rician_noise:
                new_pix=augment_rician_noise(new_pix, noise_variance=(0, 0.1))

        new_pix = new_pix.transpose((2, 0, 1)).copy()
        target = int(self.index_list[index].split('_')[-1].split('.')[0])-1
        # if np.min(new_pix)<0:
        #     print('=========error')
        return torch.from_numpy(new_pix).type(torch.FloatTensor), target      #torch.from_numpy(target).long()

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def get_split_deterministic(all_keys, fold=0, num_splits=5, random_state=12345):
    """
    Splits a list of patient identifiers (or numbers) into num_splits folds and returns the split for fold fold.
    :param all_keys:
    :param fold:
    :param num_splits:
    :param random_state:
    :return:
    """

    all_keys_sorted = np.sort(list(all_keys))
    splits = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(splits.split(all_keys_sorted)):
        if i == fold:
            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]
            break
    return train_keys, test_keys


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=2, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, labels, x):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


def main():
    args = Options().parse()


    ngpus_per_node = torch.cuda.device_count()
    # ngpus_per_node = 1
    args.world_size = ngpus_per_node * args.world_size
    args.lr = args.lr * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []


def main_worker(gpu, ngpus_per_node, args):
    # GLOBAL_SEED=23456
    # torch.manual_seed(GLOBAL_SEED)
    # torch.cuda.manual_seed(GLOBAL_SEED)
    # torch.cuda.manual_seed_all(GLOBAL_SEED)

    print('==== num_classes: ', args.num_classes)
    csvPath = args.csv_path
    train_fold_idx = args.fold


    df_tmp = pd.read_csv(csvPath)
    patients = np.array(df_tmp[df_tmp['dataset'] == 'develop']['pid'])
    train_idx, valid_idx = get_split_deterministic(patients, fold=train_fold_idx)


    args.gpu = gpu
    print('---', gpu)
    args.rank = args.rank * ngpus_per_node + gpu

    # |<--------
    # set up logger
    if args.gpu == 0:
        log_file = args.log_file
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if log_file:
            fh = logging.FileHandler(log_file)
            logger.addHandler(fh)

        logger.info('rank: {} / {}'.format(args.rank, args.world_size))
    # -------->|
    # print('rank: {} / {}'.format(args.rank, args.world_size))

    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    torch.cuda.set_device(args.gpu)
    # init the args
    global best_pred, acclist_train, acclist_val

    if args.gpu == 0:
        # print(args)
        logger.info(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
   
    trainset = DRimgDataset(index_list=train_idx,
                                    data_shape=256,
                                    label_name='label',
                                    mode='train',
                                    csv_path=csvPath,
                                    reverse=False,
                                    has_aug=True,
                            )
    valset = DRimgDataset(index_list=valid_idx,
                                    data_shape=256,
                                    label_name='label',
                                    mode='valid',
                                    csv_path=csvPath,
                                    reverse=False,
                                    has_aug=False,
                          )


    print('==== here 1')
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    print('==== here 2')
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler)

    # init the model
    model_kwargs = {}
    if args.pretrained:
        model_kwargs['pretrained'] = True
        model_kwargs['num_classes'] = args.num_classes

    if args.final_drop > 0.0:
        model_kwargs['final_drop'] = args.final_drop

    if args.dropblock_prob > 0.0:
        model_kwargs['dropblock_prob'] = args.dropblock_prob

    if args.last_gamma:
        model_kwargs['last_gamma'] = True

    if args.rectify:
        model_kwargs['rectified_conv'] = True
        model_kwargs['rectify_avg'] = args.rectify_avg

    model = encoding.models.get_model(args.model, **model_kwargs)

    print('==== here 3')
    if args.dropblock_prob > 0.0:
        from functools import partial
        from encoding.nn import reset_dropblock
        nr_iters = (args.epochs - args.warmup_epochs) * len(train_loader)
        apply_drop_prob = partial(reset_dropblock, args.warmup_epochs * len(train_loader),
                                  nr_iters, 0.0, args.dropblock_prob)
        model.apply(apply_drop_prob)

    if args.gpu == 0:
        # print(model)
        logger.info(model)

    if args.mixup > 0:
        train_loader = MixUpWrapper(args.mixup, args.num_classes, train_loader, args.gpu)
        criterion = NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        criterion = LabelSmoothing(args.label_smoothing)
    else:
        print('---------- crosssentropyloss_1 --------')
        criterion = nn.CrossEntropyLoss()

    criterion_center = CenterLoss(num_classes=2, feat_dim=2, use_gpu=True)

    print('==== here 4')
    model.cuda(args.gpu)
    criterion.cuda(args.gpu)
    criterion_center.cuda(args.gpu)
    model = DistributedDataParallel(model, device_ids=[args.gpu])
    print('==== here 5')
    # criterion and optimizer
    if args.no_bn_wd:
        parameters = model.named_parameters()
        param_dict = {}
        for k, v in parameters:
            param_dict[k] = v
        bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
        rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]
        if args.gpu == 0:
            # print(" Weight decay NOT applied to BN parameters ")
            logger.info(" Weight decay NOT applied to BN parameters ")
            # print(f'len(parameters): {len(list(model.parameters()))} = {len(bn_params)} + {len(rest_params)}')
            logger.info(f'len(parameters): {len(list(model.parameters()))} = {len(bn_params)} + {len(rest_params)}')
        optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
                                     {'params': rest_params, 'weight_decay': args.weight_decay}],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    # check point
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu == 0:
                # print("=> loading checkpoint '{}'".format(args.resume))
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1 if args.start_epoch == 0 else args.start_epoch
            best_pred = checkpoint['best_pred']
            acclist_train = checkpoint['acclist_train']
            acclist_val = checkpoint['acclist_val']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.gpu == 0:
                # print("=> loaded checkpoint '{}' (epoch {})"
                #       .format(args.resume, checkpoint['epoch']))
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                            .format(args.resume, checkpoint['epoch']))
        else:
            raise RuntimeError("=> no resume checkpoint found at '{}'". \
                               format(args.resume))
    scheduler = LR_Scheduler(args.lr_scheduler,
                             base_lr=args.lr,
                             num_epochs=args.epochs,
                             iters_per_epoch=len(train_loader),
                             warmup_epochs=args.warmup_epochs)

    def train(epoch):
        train_sampler.set_epoch(epoch)
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        global best_pred, acclist_train
        for batch_idx, (data, target) in enumerate(train_loader):
            scheduler(optimizer, batch_idx, epoch, best_pred)
            if not args.mixup:
                data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            optimizer.zero_grad()
            output = model(data)

            loss1 = criterion_center(target, output)
            loss = criterion(output, target)

            loss = 0.5 * loss + 0.5 * loss1

            loss.backward()
            optimizer.step()

            if not args.mixup:
                output = F.softmax(output, dim=1)
                acc1 = accuracy(output, target, topk=(1,))
                top1.update(acc1[0], data.size(0))

            losses.update(loss.item(), data.size(0))
            if batch_idx % 100 == 0 and args.gpu == 0:
                if args.mixup:
                    # print('Batch: %d| Loss: %.3f' % (batch_idx, losses.avg))
                    logger.info('Batch: %d| Loss: %.3f' % (batch_idx, losses.avg))
                else:
                    # print('Batch: %d| Loss: %.3f | Top1: %.3f' % (batch_idx, losses.avg, top1.avg))
                    logger.info('Batch: %d| Loss: %.3f | Top1: %.3f' % (batch_idx, losses.avg, top1.avg))

        acclist_train += [top1.avg]

    def validate(epoch):
        model.eval()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # auc = AverageMeter()
        global best_pred, acclist_train, acclist_val
        is_best = False

        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            with torch.no_grad():
                output = model(data)
                output = F.softmax(output, dim=1)
                acc1, acc5 = accuracy(output, target, topk=(1, 1))
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))

        # sum all
        sum1, cnt1, sum5, cnt5 = torch_dist_sum(args.gpu, top1.sum, top1.count, top5.sum, top5.count)

        if args.eval:
            if args.gpu == 0:
                top1_acc = sum(sum1) / sum(cnt1)
                top5_acc = sum(sum5) / sum(cnt5)
                # print('Validation: Top1: %.3f | Top5: %.3f ' % (top1_acc, top5_acc))
                logger.info('Validation: Top1: %.3f | Top5: %.3f ' % (top1_acc, top5_acc))
            return

        if args.gpu == 0:
            top1_acc = sum(sum1) / sum(cnt1)
            top5_acc = sum(sum5) / sum(cnt5)
            # print('Validation: Top1: %.3f | Top5: %.3f' % (top1_acc, top5_acc))
            logger.info('Validation: Top1: %.3f | Top5: %.3f' % (top1_acc, top5_acc))

            # save checkpoint
            acclist_val += [top1_acc]
            if top1_acc > best_pred:
                best_pred = top1_acc
                is_best = True
            encoding.utils.save_checkpoint_v2({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
                'acclist_train': acclist_train,
                'acclist_val': acclist_val,
            }, args=args, is_best=is_best)


    if args.export:
        if args.gpu == 0:
            torch.save(model.module.state_dict(), args.export + '.pth')
        return

    if args.eval:
        validate(args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs):
        tic = time.time()
        train(epoch)
        if epoch % 1 == 0:  # or epoch == args.epochs-1:
            validate(epoch)
        elapsed = time.time() - tic
        if args.gpu == 0:
            # print(f'Epoch: {epoch}, Time cost: {elapsed}')
            logger.info(f'Epoch: {epoch}, Time cost: {elapsed}')

    if args.gpu == 0:
        encoding.utils.save_checkpoint({
            'epoch': args.epochs - 1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'acclist_train': acclist_train,
            'acclist_val': acclist_val,
        }, args=args, is_best=False)


if __name__ == "__main__":
    main()
