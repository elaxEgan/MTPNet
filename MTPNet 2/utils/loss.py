import torch
import torch.nn.functional as F
import torch.nn as nn


def get_contour(pre):
    lbl = pre.gt(0.5).float()
    ero = 1 - F.max_pool2d(1 - lbl, kernel_size=5, stride=1, padding=2)  # erosion
    dil = F.max_pool2d(lbl, kernel_size=5, stride=1, padding=2)  # dilation

    edge = dil - ero
    return edge


class SEALoss(nn.Module):
    def __init__(self, radius, num_modal=1):
        super(SEALoss, self).__init__()
        self.radius = radius
        self.num_modal = num_modal
        self.diameter = 2 * radius + 1
        self.avg_pool = nn.AvgPool2d(self.diameter, stride=1, padding=radius)

    def forward(self, pred, label):
        pred = pred.float()
        sal_map = F.interpolate(pred, scale_factor=1, mode='bilinear', align_corners=True)
        label_ = F.interpolate(label, size=sal_map.shape[-2:], mode='bilinear', align_corners=True)

        mask = get_contour(sal_map) + 1e-10

        avg_features = self.avg_pool(label_)
        avg_pred = self.avg_pool(sal_map)

        alignment = 1 - F.cosine_similarity(avg_features, avg_pred, dim=1, eps=1e-8)
        dis_sal = torch.abs(avg_features[:, -1:, :, :])
        distance = dis_sal * alignment

        loss = distance
        loss = torch.sum(loss * mask) / torch.sum(mask)

        return loss


