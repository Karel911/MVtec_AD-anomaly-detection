import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataloader import get_train_augmentation, get_test_augmentation, get_loader
from util.utils import AvgMeter
from util.losses import Optimizer, Scheduler, Criterion, arcFace
from model.EfficientNet import EfficientNet
from sklearn.metrics import f1_score
import timm


class Trainer():
    def __init__(self, args, save_path, fold=None, unique_label=None, tr_gt=None, null_value=None):
        super(Trainer, self).__init__()
        self.args = args
        self.save_path = save_path
        self.fold = fold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.size = args.img_size

        self.target_class = unique_label
        self.num_classes = len(np.unique(tr_gt))

        self.null_value = null_value
        self.tr_img_folder = os.path.join(args.data_path, 'train/')
        self.tr_gt = tr_gt

        self.train_transform = get_train_augmentation(img_size=args.img_size, ver=args.aug_ver)
        self.test_transform = get_test_augmentation(img_size=args.img_size)

        self.train_loader = get_loader(self.args, self.tr_img_folder, self.tr_gt, phase='train', fold=fold,
                                       batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                       transform=self.train_transform)
        self.val_loader = get_loader(self.args, self.tr_img_folder, self.tr_gt, phase='val', fold=fold,
                                     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                     transform=self.train_transform)

        # Network
        if self.args.model == 'efficientnet':
            self.model = EfficientNet.from_pretrained(f'efficientnet-b{args.arch}', advprop=True,
                                                      num_classes=self.num_classes).to(self.device)
        else:
            self.model = timm.create_model(self.args.model_name, pretrained=True, num_classes=self.num_classes).to(self.device)

        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)

        # Loss and Optimizer
        self.criterion = Criterion(args)
        self.optimizer = Optimizer(args, self.model)
        self.scheduler = Scheduler(args, self.optimizer)

    def training(self, args):
        self.model.train()
        train_loss = AvgMeter()
        preds = []
        GTs = []

        for images, gts in tqdm(self.train_loader):
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            gts = torch.tensor(gts, device=self.device, dtype=torch.long)
            self.optimizer.zero_grad()
            outputs = self.model(images)

            if self.args.arcloss == 'arcface':
                if self.args.train_method == 'one_class':
                    outputs = arcFace(self.args, outputs['norm_output'], gts, num_classes=self.num_classes,
                                      null_value=self.null_value)
                else:
                    outputs = arcFace(self.args, outputs['norm_output'], gts)

            loss = self.criterion(outputs, gts)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)
            self.optimizer.step()

            preds += torch.softmax(outputs, dim=-1).argmax(1).detach().cpu().numpy().tolist()
            GTs += gts.detach().cpu().numpy().tolist()

            # log
            train_loss.update(loss.item(), n=images.size(0))

        # Metric
        train_f1 = f1_score(GTs, preds, average='macro')

        print(f'Epoch:[{self.epoch:03d}/{args.epochs:03d}]')
        print(f'Train Loss:{train_loss.avg:.3f} | F1_score:{train_f1:.3f}')

        return train_loss.avg, train_f1

    def validate(self):
        self.model.eval()
        val_loss = AvgMeter()
        preds = []
        GTs = []

        with torch.no_grad():
            for images, gts in tqdm(self.val_loader):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                gts = torch.tensor(gts, device=self.device, dtype=torch.long)
                outputs = self.model(images)

                if self.args.arcloss == 'arcface':
                    loss = self.criterion(outputs['inference'], gts)
                    preds += torch.softmax(outputs['inference'], dim=-1).argmax(1).detach().cpu().numpy().tolist()
                else:
                    loss = self.criterion(outputs, gts)
                    preds += torch.softmax(outputs, dim=-1).argmax(1).detach().cpu().numpy().tolist()

                GTs += gts.detach().cpu().numpy().tolist()

                # log
                val_loss.update(loss.item(), n=images.size(0))

        # Metric
        val_f1 = f1_score(GTs, preds, average='macro')

        print(f'Valid Loss:{val_loss.avg:.3f} | F1_score:{val_f1:.3f}')
        return val_loss.avg, val_f1, GTs, preds

    def init(self):
        # Train / Validate
        min_loss = 1000
        early_stopping = 0
        best_f1 = 0
        t = time.time()
        for epoch in range(1, self.args.epochs + 1):
            self.epoch = epoch
            train_loss, train_f1 = self.training(self.args)

            if self.args.train_method == 'one_class':
                if self.args.scheduler == 'Reduce':
                    self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()
            else:
                val_loss, val_f1, _, _ = self.validate()
                if self.args.scheduler == 'Reduce':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save models
            if self.args.train_method == 'one_class':
                if train_f1 == 1.0:
                    min_loss = train_loss
                    best_f1 = train_f1
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, f'model_{epoch}.pth'))
                    print(f'-----------------SAVE:{best_epoch}epoch----------------')
                    break
                elif (train_loss < min_loss) and (best_f1 < train_f1):
                    best_epoch = epoch
                    best_f1 = train_f1
                    min_loss = train_loss
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, f'model_{epoch}.pth'))
                    print(f'-----------------SAVE:{best_epoch}epoch----------------')
                else:
                    early_stopping += 1
            else:
                if val_loss < min_loss:
                    early_stopping = 0
                    best_epoch = epoch
                    best_f1 = val_f1
                    min_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, f'model_{epoch}.pth'))
                    print(f'-----------------SAVE:{best_epoch}epoch----------------')
                else:
                    early_stopping += 1

            if early_stopping == self.args.patience:
                break

        print(f'\nBest Val Epoch:{best_epoch} | Val Loss:{min_loss:.3f} | Val F1_score:{best_f1:.3f} '
              f'time: {(time.time() - t) / 60:.3f}M')

        end = time.time()
        print(f'Total Process time:{(end - t) / 60:.3f}Minute')

        return min_loss, best_f1


class Tester():
    def __init__(self, args, save_path, num_classes=None):
        super(Tester, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_transform = get_test_augmentation(img_size=args.img_size)
        self.save_path = save_path
        self.num_classes = num_classes

        # Network
        if self.args.model == 'efficientnet':
            self.model = EfficientNet.from_pretrained(f'efficientnet-b{args.arch}', advprop=True,
                                                      num_classes=self.num_classes).to(self.device)
        else:
            self.model = timm.create_model(self.args.model_name, pretrained=True, num_classes=self.num_classes).to(self.device)

        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)

        self.model.load_state_dict(torch.load(save_path))
        print('###### pre-trained Model restored #####')

        te_img_folder = os.path.join(args.data_path, 'test/')
        self.test_loader = get_loader(self.args, te_img_folder, gt_folder=None, phase='test', fold=None,
                                      batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                      transform=self.test_transform)

    def test(self):
        self.model.eval()
        pred = torch.FloatTensor().cuda()

        with torch.no_grad():
            for i, images in enumerate(tqdm(self.test_loader)):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                outputs = self.model(images)
                if self.args.arcloss == 'arcface':
                    pred = torch.cat([pred, outputs['inference']])
                else:
                    pred = torch.cat([pred, outputs])

        if self.args.ensemble:
            return pred
