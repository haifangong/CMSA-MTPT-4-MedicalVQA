import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def CELoss(logit, target, reduction='mean'):
    criterion = nn.CrossEntropyLoss(reduction=reduction)
    return criterion(logit, target)

def cal_acc(y_pred, y):
    y_pred = torch.argmax(y_pred, dim=1, keepdim=False)
    return torch.sum(y_pred == y).float() / y.shape[0]

def cal_iou(pred, gt, n_classes):
    pred = torch.argmax(pred, dim=1)
    assert len(pred.shape) == 3
    total_iou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]
        # print(pred_tmp.max(), pred_tmp.min())
        # print(gt_tmp.max(), gt_tmp.min())
        # print("pred_tmp.shape: {}".format(pred_tmp.shape))
        # print("gt_tmp.shape: {}".format(gt_tmp.shape))

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            a = (pred_tmp == j) 
            b = (gt_tmp == j)
            # print((a==b).all())

            match = (pred_tmp == j).float() + (gt_tmp == j).float()

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un
        # print("intersect: {}".format(intersect))
        # print("union: {}".format(union))
        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])
        # print("iou: {}".format(iou))
        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou
        # print(img_iou)

    return total_iou

def get_segment_labels():
    return np.array([
        [  0,   0,   0], 
        [128,   0,   0], 
        [  0, 128,   0], 
        [128, 128,   0],
        [  0,   0, 128], 
        [128,   0, 128], 
        [  0, 128, 128], 
        [128, 128, 128],
        [ 64,   0,   0], 
        [192,   0,   0], 
        [ 64, 128,   0], 
        [192, 128,   0],
        [ 64,   0, 128], 
        [192,   0, 128], 
        [ 64, 128, 128], 
        [192, 128, 128],
        [  0,  64,   0], 
        [128,  64,  0]])

def decode_segmap(label_mask):
    n_classes = 18
    label_colours = get_segment_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def decode_seg_map_sequence(label_masks):
    assert(label_masks.ndim == 3 or label_masks.ndim == 4)
    if label_masks.ndim == 4:
        label_masks = label_masks.squeeze(1)
    assert(label_masks.ndim == 3)
    rgb_masks = []
    for i in range(label_masks.shape[0]):
        rgb_mask = decode_segmap(label_masks[i])
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)