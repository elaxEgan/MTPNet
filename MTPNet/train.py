import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb, os, argparse
from datetime import datetime
from model.MTPNet import MTPNet
from utils.data import get_loader
from utils.func import label_edge_prediction, AvgMeter, clip_gradient, adjust_lr
import pytorch_iou
import pytorch_fm
from utils.loss import SEALoss

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=70, help='epoch number')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--encoder_dim', default=[96, 192, 384, 768], type=int, help='dim of each encoder layer')
parser.add_argument('--feature_size', default=[176, 88, 44, 22, 11], type=list, help='the size of feature')
parser.add_argument('--dim', default=64, type=int, help='dim')
parser.add_argument('--embed_dim', default=384, type=int, help='embedding dim')
parser.add_argument('--task_num', default=[1, 2, 5, 10], type=list, help='the number of task prompt in encoder')
parser.add_argument('--pretrained_model', default='./pretrained_model/swin_tiny_patch4_window7_224.pth', type=str,
                    help='load Pretrained model')

opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
# build models
model = MTPNet(args=opt)


model.cuda()

params = model.parameters()

optimizer = torch.optim.Adam(params, opt.lr)

image_root = '/home/dell/HJL/remote/datasets/EORSSD/train-images/'
gt_root = '/home/dell/HJL/remote/datasets/EORSSD/train-labels/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average=True)
floss = pytorch_fm.FLoss()
size_rates = [1]  # multi-scale training
SEA = SEALoss(radius=5)
loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]

print(next(model.parameters()).is_cuda)


def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # edge prediction
            edges = label_edge_prediction(gts)

            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)

            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                edges = F.interpolate(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            outputs_saliency, outputs_edge, outputs_saliency_s, outputs_edge_s, task_prompt, q = model(
                images)

            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
            edg_1_16, edg_1_8, edg_1_4, edg_1_1 = outputs_edge
            mask_1_16_s, mask_1_8_s, mask_1_4_s, mask_1_1_s = outputs_saliency_s
            edg_1_16_s, edg_1_8_s, edg_1_4_s, edg_1_1_s = outputs_edge_s

            size_1_16, size_1_8, size_1_4, size_1_1 = mask_1_16.size()[2:], \
                mask_1_8.size()[2:], \
                mask_1_4.size()[2:], \
                mask_1_1.size()[2:]
            label_s_1_16, label_s_1_8, label_s_1_4, label_s_1_1 = F.interpolate(gts, size=size_1_16, mode='bilinear',
                                                                                align_corners=True), \
                F.interpolate(gts, size=size_1_8, mode='bilinear', align_corners=True), \
                F.interpolate(gts, size=size_1_4, mode='bilinear', align_corners=True), \
                F.interpolate(gts, size=size_1_1, mode='bilinear', align_corners=True)
            label_s_1_16, label_s_1_8, label_s_1_4, label_s_1_1 = Variable(label_s_1_16).cuda(), \
                Variable(label_s_1_8).cuda(), \
                Variable(label_s_1_4).cuda(), \
                Variable(label_s_1_1).cuda()

            label_e_1_16, label_e_1_8, label_e_1_4, label_e_1_1 = F.interpolate(edges, size=size_1_16,
                                                                                mode='bilinear', align_corners=True), \
                F.interpolate(edges, size=size_1_8,
                              mode='bilinear', align_corners=True), \
                F.interpolate(edges, size=size_1_4,
                              mode='bilinear', align_corners=True), \
                F.interpolate(edges, size=size_1_1,
                              mode='bilinear', align_corners=True)
            label_e_1_16, label_e_1_8, label_e_1_4, label_e_1_1 = Variable(label_e_1_16).cuda(), \
                Variable(label_e_1_8).cuda(), \
                Variable(label_e_1_4).cuda(), \
                Variable(label_e_1_1).cuda()

            # saliency loss
            loss5_1 = CE(mask_1_16, label_s_1_16)
            loss4_1 = CE(mask_1_8, label_s_1_8)
            loss3_1 = CE(mask_1_4, label_s_1_4)
            loss1_1 = CE(mask_1_1, label_s_1_1)
            loss1_i = IOU(torch.sigmoid(mask_1_1), label_s_1_1)
            # loss2_i = IOU(torch.sigmoid(mask_1_4), gts)
            # loss3_i = IOU(torch.sigmoid(mask_1_8), gts)
            # loss4_i = IOU(torch.sigmoid(mask_1_16), gts)
            loss1_BTM = SEA(torch.sigmoid(mask_1_1), label_s_1_1)

            # edge loss
            c_loss5 = CE(edg_1_16, label_e_1_16)
            c_loss4 = CE(edg_1_8, label_e_1_8)
            c_loss3 = CE(edg_1_4, label_e_1_4)
            c_loss1 = CE(edg_1_1, label_e_1_1)

            # saliency loss
            loss5_s = CE(mask_1_16_s, label_s_1_16)
            loss4_s = CE(mask_1_8_s, label_s_1_8)
            loss3_s = CE(mask_1_4_s, label_s_1_4)
            loss1_s = CE(mask_1_1_s, label_s_1_1)
            loss1_s_i = IOU(torch.sigmoid(mask_1_1_s), label_s_1_1)
            loss1_s_BTM = SEA(torch.sigmoid(mask_1_1_s), label_s_1_1)

            # edge loss
            c_loss5_s = CE(edg_1_16_s, label_e_1_16)
            c_loss4_s = CE(edg_1_8_s, label_e_1_8)
            c_loss3_s = CE(edg_1_4_s, label_e_1_4)
            c_loss1_s = CE(edg_1_1_s, label_e_1_1)


            task_prompt_loss = torch.log(torch.abs(
                F.cosine_similarity(task_prompt[:, :18, :].reshape(-1), task_prompt[:, 18:, :].reshape(-1), dim=0)) + 1)

            diff = task_prompt[:, :18, :] - task_prompt[:, 18:, :]
            l2_distance_squared = (diff ** 2).sum(dim=2)

            cosloss = task_prompt_loss + 1/l2_distance_squared.mean()

            img_total_loss = loss_weights[0] * loss1_1 + loss_weights[2] * loss3_1 + loss_weights[3] * loss4_1 + \
                             loss_weights[
                                 4] * loss5_1
            edge_total_loss = loss_weights[0] * c_loss1 + loss_weights[2] * c_loss3 + loss_weights[3] * c_loss4 + \
                                 loss_weights[4] * c_loss5
            img_total_loss_s = loss_weights[0] * loss1_s + loss_weights[2] * loss3_s + loss_weights[3] * loss4_s + \
                               loss_weights[4] * loss5_s
            edge_total_loss_s = loss_weights[0] * c_loss1_s + loss_weights[2] * c_loss3_s + loss_weights[
                3] * c_loss4_s + loss_weights[4] * c_loss5_s

            # img_total_loss_i = loss_weights[0] * loss1_i +  loss_weights[1] * loss2_i +  loss_weights[2] * loss3_i +  loss_weights[3] * loss4_i
            img_total_loss_i = loss_weights[0] * loss1_i
            img_total_loss_s_i = loss_weights[0] * loss1_s_i

            total_loss = img_total_loss + edge_total_loss + img_total_loss_s + edge_total_loss_s + img_total_loss_i + img_total_loss_s_i + cosloss + loss1_BTM + loss1_s_BTM

            loss = total_loss

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1:
                loss_record1.update(img_total_loss.data, opt.batchsize)
                loss_record2.update(edge_total_loss.data, opt.batchsize)
                loss_record3.update(img_total_loss_s.data, opt.batchsize)
                loss_record4.update(edge_total_loss_s.data, opt.batchsize)
                loss_record5.update(cosloss.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step,
                       opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, img_total_loss.data,
                       edge_total_loss.data))

    save_path = 'models/MTPNet/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch + 1) >= 30:
        torch.save(model.state_dict(), save_path + 'MTPNet_EORSSD.pth' + '.%d' % epoch)


print("Let's go!")
if __name__ == '__main__':
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)