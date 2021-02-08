# coding:utf-8
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: TXVision
## Email: cweidao@infervion.com
## Copyright (c) 2021
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import time
import cv2
import random

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import encoding
from encoding.utils import (accuracy, AverageMeter, MixUpWrapper, LR_Scheduler)

from decimal import Decimal

class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='Inference')
        parser.add_argument('--dataset', type=str, default='imagenet',
                            help='training dataset (default: imagenet)')
        parser.add_argument('--num_classes', type=int, metavar='N',
                            help='num_classes')
        parser.add_argument('--fold', type=int, metavar='N',
                            help='the fold of K-Fold')
        parser.add_argument('--csv-path', type=str,
                            help='the csv file contained all clinical info of pids')
        parser.add_argument('--base-size', type=int, default=None,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=224,
                            help='crop image size')
        # model params
        parser.add_argument('--model', type=str, default='densenet',
                            help='network model type (default: densenet)')
        parser.add_argument('--rectify', action='store_true',
                            default=False, help='rectify convolution')
        parser.add_argument('--rectify-avg', action='store_true',
                            default=False, help='rectify convolution')
        # training hyper params
        parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                            help='batch size for training (default: 128)')
        parser.add_argument('--workers', type=int, default=4,
                            metavar='N', help='dataloader threads')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true',
                            default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str,  # default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--verify', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--export', type=str, default=None,
                            help='put the path to resuming file if needed')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args


def draw_roc(ensemble_gt, ensemble_pred, name='en'):
    import matplotlib.pyplot as plt
    # 计算置信区间
    def ciauc(auc, pos_n, neg_n):
        import math
        q0 = auc * (1 - auc)
        q1 = auc / (2 - auc) - auc ** 2
        q2 = 2 * (auc ** 2) / (1 + auc) - auc ** 2
        se = math.sqrt((q0 + (pos_n - 1) * q1 + (neg_n - 1) * q2) / (pos_n * neg_n))
        z_crit = 1.959964
        lower = auc - z_crit * se
        upper = auc + z_crit * se
        lower = max(lower, 0.)
        upper = min(upper, 1.)
        # print("[{:.3f}, {:.3f}]".format(lower, upper))
        return lower, upper

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    num_pos, num_neg = np.sum(ensemble_gt), len(ensemble_gt) - np.sum(ensemble_gt)
    fpr, tpr, threshold = roc_curve(np.array(ensemble_gt), np.array(ensemble_pred))  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    lower, upper = ciauc(roc_auc, num_pos, num_neg)
    plt.plot(fpr, tpr, color='darkorange', alpha=.8,
             lw=lw, label='Ensemble (AUC:{:0.3f} [95%CI, {:0.3f}-{:0.3f}])'.format(roc_auc, lower,
                                                                                   upper))  ###假正率为横坐标，真正率为纵坐标做曲线

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity', size=18, weight='bold')
    plt.ylabel('Sensitivity', size=18, weight='bold')
    plt.title('Testing ROC curves')
    plt.legend(loc="lower right")
    plt.savefig(name + '_roc.jpg')
    plt.show()


def get_2D_gaussian_map(im_h, im_w):
    from numpy import matlib as mb
    IMAGE_WIDTH = im_w
    IMAGE_HEIGHT = im_h

    center_x = IMAGE_WIDTH / 2
    center_y = IMAGE_HEIGHT / 2

    R = np.sqrt(center_x ** 2 + center_y ** 2)

    # Gauss_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    # 直接利用矩阵运算实现
    mask_x = mb.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)
    mask_y = mb.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)

    x1 = np.arange(IMAGE_WIDTH)
    x_map = mb.repmat(x1, IMAGE_HEIGHT, 1)

    y1 = np.arange(IMAGE_HEIGHT)
    y_map = mb.repmat(y1, IMAGE_WIDTH, 1)
    y_map = np.transpose(y_map)
    Gauss_map = np.sqrt((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2)
    Gauss_map = np.exp(-0.5 * Gauss_map / R)
    return Gauss_map


def multiScaleSharpen_v1(img, radius=5):
    img = np.float32(img)
    Dest_float_img = np.zeros(img.shape, dtype=np.float32) + 114

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
        self.data_shape = data_shape
        self.reverse = reverse
        self.label_name = label_name
        self.index_list = index_list
        self.add_gaussian_mask = True
        self.add_edge = True
        self.detail_enhancement = True
        self.wavelet_trans = True
        self.padding = True
        self.resize = True
        self.mosaic = True
        self.has_aug = has_aug
        self.random_rotate = True
        self.random_lightness = True
        self.random_transpose = True
        self.random_mirror = True
        self.random_brightness = False
        self.random_gaussian_noise = False
        self.random_rician_noise = False
        self.len = len(index_list)
        self.all_df = pd.read_csv(csv_path)

        print('=== mode:' + self.mode)
        print('=== num of samples: ', self.len)
        print('=== num of l1 samples: ',
              len([item for item in self.index_list if int(item.split('_')[-1].split('.')[0]) == 1]))
        print('=== num of l2 samples: ',
              len([item for item in self.index_list if int(item.split('_')[-1].split('.')[0]) == 2]))

    def load_mosaic(self, index):
        # loads images in a mosaic
        s = self.data_shape
        xc, yc = [int(random.uniform(s * 0.75, s * 1.25)) for _ in range(2)]  # mosaic center x, y
        ll_ = int(index.split('_')[-1].split('.')[0])
        s_list = [item for item in self.index_list if int(item.split('_')[-1].split('.')[0]) == ll_]
        # s_list = list(self.all_df[(self.all_df['dataset'] == 'develop') & (self.all_df['label'] == ll_)]['pid'])

        np.random.shuffle(s_list)
        indices = [index] + s_list[:3]  # 3 additional image indices
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
            new_pix = np.zeros((max_edge, max_edge, num_channel), dtype=np.uint8) + 114
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
        new_pix, _, _ = self.load_image_0(file_path)
        new_pix = new_pix.transpose((2, 0, 1)).copy()
        target = int(self.index_list[index].split('_')[-1].split('.')[0]) - 1
        if self.mode == 'inference':
            return torch.from_numpy(new_pix).type(torch.FloatTensor), target, self.index_list[index]
        return torch.from_numpy(new_pix).type(torch.FloatTensor), target  # torch.from_numpy(target).long()

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


def main():
    # init the args
    args = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    csvPath = args.csv_path
    train_fold_idx = args.fold
    
    df_tmp = pd.read_csv(csvPath)
    valid_idx = np.array(df_tmp[df_tmp['dataset'] == 'test']['pid'])

    valset = DRimgDataset(index_list=valid_idx,
                          data_shape=256,
                          label_name='label',
                          mode='inference',
                          csv_path=csvPath,
                          reverse=False,
                          has_aug=False,
                          )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True if args.cuda else False)

    # init the model
    model_kwargs = {'pretrained': True}
    model_kwargs['num_classes'] = args.num_classes

    if args.rectify:
        model_kwargs['rectified_conv'] = True
        model_kwargs['rectify_avg'] = args.rectify_avg

    model = encoding.models.get_model(args.model, **model_kwargs)

    print(model)

    if args.cuda:
        # torch.cuda.set_device(0)
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = nn.DataParallel(model)

    # checkpoint
    if args.verify:
        if os.path.isfile(args.verify):
            print("=> loading checkpoint '{}'".format(args.verify))
            model.module.load_state_dict(torch.load(args.verify))
        else:
            raise RuntimeError("=> no verify checkpoint found at '{}'". \
                               format(args.verify))
    elif args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            raise RuntimeError("=> no resume checkpoint found at '{}'". \
                               format(args.resume))

    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    res_dict = {}
    is_best = False
    tbar = tqdm(val_loader, desc='\r')
    for batch_idx, (data, target, pid_) in enumerate(tbar):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            output = F.softmax(output, dim=1)
            res_dict[batch_idx] = [target.cpu().numpy(), output.cpu().numpy(), pid_]
            # for accuracy func, output must be one-hot style
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

        tbar.set_description('Top1: %.3f | Top5: %.3f' % (top1.avg, top5.avg))

    print('Top1 Acc: %.3f | Top5 Acc: %.3f ' % (top1.avg, top5.avg))

    pid_list_tmp = []
    y_true_tmp = []
    y_pred_tmp = []
    for k, v in res_dict.items():
        b_pid_list = v[-1]
        b_y_true = v[0]
        b_y_pred = v[1]
        for i in range(len(b_pid_list)):
            pid_list_tmp.append(b_pid_list[i])
            y_true_tmp.append(b_y_true[i])
            y_pred_tmp.append(b_y_pred[i, 1])
    res_name = args.resume.split('/')[-1] + '_' + str(args.fold).zfill(2) + '_res.csv'
    pd.DataFrame({'pid': pid_list_tmp, 'y_true': y_true_tmp, 'y_pred': y_pred_tmp}).to_csv(res_name)

    print('auc_old: ', round(roc_auc_score(np.array(y_true_tmp), np.array(y_pred_tmp)), 3))
    print('auc_fresh: ', round(roc_auc_score(1.0 - np.array(y_true_tmp), 1.0 - np.array(y_pred_tmp)), 3))

    im_name = args.resume.split('/')[-1] + '_' + str(train_fold_idx)
    draw_roc(y_true_tmp, y_pred_tmp, im_name)

    if args.export:
        torch.save(model.module.state_dict(), args.export + '.pth')


if __name__ == "__main__":
    main()
