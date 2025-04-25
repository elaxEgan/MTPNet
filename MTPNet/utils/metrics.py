import numpy as np

precisions = []
recalls = []
thresholds = np.arange(256) / 255  # 阈值从0到255

def calculate_precision_recall(gt, res):
    for threshold in thresholds:

        binary_res = (res > threshold).astype(int)
        binary_gt = (gt > threshold).astype(int)


        TP = np.sum((binary_gt == 1) & (binary_res == 1))
        FP = np.sum((binary_gt == 0) & (binary_res == 1))
        FN = np.sum((binary_gt == 1) & (binary_res == 0))
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


def calculate_mae(S, GT):
    H, W = S.shape
    mae = (1 / (W * H)) * np.sum(np.abs(S - GT))
    return mae
