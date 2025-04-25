# -*-ding:utf-8-*-
import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
# from scipy import misc
import time
import imageio
from model.MTPNet import MTPNet
from utils.data import test_dataset2

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
argument = parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--encoder_dim', default=[96,192,384,768], type=int, help='dim of each encoder layer')
parser.add_argument('--task_num', default=[1,2,5,10], type=list, help='the number of task prompt in encoder')
parser.add_argument('--feature_size', default=[176, 88, 44, 22, 11], type=list, help='the size of feature')
parser.add_argument('--dim', default=64, type=int, help='dim')
parser.add_argument('--embed_dim', default=384, type=int, help='embedding dim')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--pretrained_model', default='./pretrained_model/swin_tiny_patch4_window7_224.pth', type=str, help='load Pretrained model')
args = parser.parse_args()
opt = args

dataset_path = '/home/dell/HJL/remote/datasets/'

model = MTPNet(opt)
model.load_state_dict(torch.load('./models/MTPNet/MTPNet_EORSSD.pth.1'))

model.cuda()
model.eval()

test_datasets = ['EORSSD']

precisions = []
recalls = []
precision_list = []
recall_list = []
precision_total = []
recall_total = []
mae_total = []

thresholds = np.arange(256) / 255  # 阈值从0到255


# thresholds = [0.5]


def calculate_precision_recall(gts, res):
    for threshold in thresholds:
        for idx in range(len(gts)):
            gt = gts[idx]
            res_single = res[idx]

            binary_res = (res_single > threshold).astype(int)
            binary_gt = (gt > threshold).astype(int)

            TP = np.sum((binary_gt == 1) & (binary_res == 1))
            FP = np.sum((binary_gt == 0) & (binary_res == 1))
            FN = np.sum((binary_gt == 1) & (binary_res == 0))
            precision = TP / (TP + FP + 1e-10)
            recall = TP / (TP + FN + 1e-10)

            precisions.append(precision)
            recalls.append(recall)
    return precisions, recalls



def calculate_fbeta(precisions, recalls, beta):
    fbeta_list = []
    for precision, recall in zip(precisions, recalls):
        if precision + recall != 0:
            fbeta = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
            fbeta_list.append(fbeta)
        else:
            fbeta_list.append(0.0)
    return fbeta_list



def calculate_mae(S_list, GT_list):
    mae_list = []
    for S, GT in zip(S_list, GT_list):
        H, W = S.shape
        mae = (1 / (W * H)) * np.sum(np.abs(S - GT))
        mae_list.append(mae)
    return mae_list



for dataset in test_datasets:
    save_path = './results/' + 'vis-' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/test-images/'
    gt_root = dataset_path + dataset + '/test-labels/'
    test_loader = test_dataset2(image_root, gt_root, opt.testsize, opt.batchsize)
    time_sum = 0
    # Calculate the number of batches
    num_batches = int(np.ceil(test_loader.size / test_loader.batch_size))

    for i in range(num_batches):
        images, gts, names = test_loader.load_data()
        images = torch.stack(images).cuda()  # Stack images into a batch
        gt = [np.asarray(gt, np.float32) for gt in gts]
        gt = [g / (g.max() + 1e-8) for g in gt]

        time_start = time.time()
        outputs_saliency, outputs_edg, outputs_saliency_s, outputs_edg_s, task_prompt, all_dict = model(
            images)
        time_end = time.time()
        time_sum = time_sum + (time_end - time_start)

        mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
        mask_1_16_s, mask_1_8_s, mask_1_4_s, mask_1_1_s = outputs_saliency_s

        res = mask_1_1_s
        # Interpolate each res to match the size of corresponding gt
        res = [F.interpolate(r.unsqueeze(0), size=gt[i].shape, mode='bilinear', align_corners=False) for i, r in
               enumerate(res)]
        res = [r.sigmoid().data.cpu().numpy().squeeze() for r in res]
        res = [(r - r.min()) / (r.max() - r.min() + 1e-8) for r in res]

        # Here you can save or process the result for the current batch

        precisions, recalls = calculate_precision_recall(gt, res)
        fbeta_list = calculate_fbeta(precisions, recalls, 0.3)
        fbeta_mean = sum(fbeta_list) / len(fbeta_list)

        fbeta_total = []
        fbeta_total.append(fbeta_mean)

        mae = calculate_mae(res, gt)
        mae_total.extend(mae)

        # Convert the result to uint8 type before saving
        res = [(r * 255).astype(np.uint8) for r in res]
        for r, name in zip(res, names):
            imageio.imsave(save_path + name, r)

    # Calculate average metrics after the loop
    print('Running time {:.5f}'.format(time_sum / test_loader.size))
    print('Average speed: {:.4f} fps'.format(test_loader.size / time_sum))

    fbeta_average = sum(fbeta_total) / len(fbeta_total)
    print("Average fbeta:", fbeta_average)

    mae_average = sum(mae_total) / len(mae_total)
    print("Average MAE:", mae_average)