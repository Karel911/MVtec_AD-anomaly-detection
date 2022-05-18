import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from config import getConfig

args = getConfig()


def Optimizer(args, model):
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    return optimizer


def Scheduler(args, optimizer):
    if args.scheduler == 'Reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, patience=args.patience)
    elif args.scheduler == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.9)
        
    return scheduler


def Criterion(args):
    if args.criterion == 'smoothing':
        criterion = LabelSmoothingLoss(88, smoothing=0.1, dim=-1)
    elif args.criterion == 'ce':
        criterion = torch.nn.CrossEntropyLoss()

    return criterion


def arcFace(args, norm_input, gts, num_classes=None, null_value=None):
    """
    model output : (N, dim)
    normalized weights : (dim, n_classes)

    bottle_anomaly idx : 0 ~ 3, where
    cable_anomaly idx : 4 ~ 12, where normal (9)
    capsule_anomaly idx : 13 ~ 18, where normal (15)
    pill_anomaly idx : 47 ~ 54, where normal (52)
    screw_anomaly idx : 55 ~ 60, where normal (55)
    transistor_anomaly idx : 69 ~ 73, where normal (72)
    zipper_anomaly idx : 80 ~ 87, where normal (84)
    """

    cos_theta = norm_input.clamp(-1, 1)
    theta = cos_theta.acos_()

    # Discriminate features near the decision boundary
    if args.train_method == 'ysw':
        target_classes = torch.LongTensor(
            [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 47, 48, 49, 50, 51, 53, 54, 56,
             57, 58, 59, 60, 69, 70, 71, 73, 80, 81, 82, 83, 85, 86, 87]).cuda()
        one_hot = torch.zeros(gts.size()[0], 88).cuda()
    elif args.train_method == 'one_class':
        unique = torch.unique(gts)
        target_classes = unique[unique != null_value].cuda()
        one_hot = torch.zeros(gts.size()[0], num_classes).cuda()
    else:
        target_classes = torch.LongTensor(
            [4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 56,
             57, 58, 59, 60, 69, 70, 71, 73, 80, 81, 82, 83, 85, 86, 87]).cuda()
        one_hot = torch.zeros(gts.size()[0], 88).cuda()

    for i, c in enumerate(gts):
        if c in target_classes:
            one_hot[i, c] += args.margin

    theta += one_hot
    final_logit = theta.cos_().mul_(args.scale)

    return final_logit


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
