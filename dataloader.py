import cv2
import glob
import torch
import numpy as np
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from torch.utils.data import Dataset, DataLoader
from config import getConfig

args = getConfig()

class DatasetGenerate(Dataset):
    def __init__(self, args, img_folder, tr_gt, phase: str = 'train', fold=None, transform=None):
        self.images = sorted(glob.glob(img_folder + '/*.png'))
        self.gts = tr_gt
        self.transform = transform

        if args.train_method == 'one_class':
            pass
        else:
            self.idx = np.load(f'data/{args.fold}-Fold_idx.npy', allow_pickle=True)[fold - 1]
            tr_idx, val_idx = self.idx

            if phase == 'train':
                self.images, self.gts = np.array(self.images)[tr_idx], np.array(self.gts)[tr_idx]
            elif phase == 'val':
                self.images, self.gts = np.array(self.images)[val_idx], np.array(self.gts)[val_idx]
            else:  # Test
                pass

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = self.gts[idx]

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, gt

    def __len__(self):
        return len(self.images)


class Test_DatasetGenerate(Dataset):
    def __init__(self, img_folder, transform=None):
        self.images = sorted(glob.glob(img_folder + '/*.png'))
        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image

    def __len__(self):
        return len(self.images)


def get_loader(args, img_folder, gt_folder, phase: str, fold, batch_size, shuffle, num_workers, transform):
    if phase == 'test':
        dataset = Test_DatasetGenerate(img_folder, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        dataset = DatasetGenerate(args, img_folder, gt_folder, phase, fold, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                 drop_last=True)

    print(f'{phase} length : {len(dataset)}')

    return data_loader


def get_train_augmentation(img_size, ver):
    if ver == 1:
        transforms = albu.Compose([
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    if ver == 2:
        transforms = albu.Compose([
            albu.OneOf([
                albu.RandomRotate90(),
                albu.Rotate(limit=45)
            ], p=0.7),
            albu.OneOf([
                albu.VerticalFlip()
                # albu.HorizontalFlip()
            ], p=0.5),
            ShiftScaleRotate(shift_limit=0.0625,
                             scale_limit=0.1,
                             rotate_limit=45,
                             p=1.0, always_apply=True),
            albu.OneOf([
                albu.RandomContrast(),
                albu.RandomGamma(),
                albu.RandomBrightness(),
            ], p=0.5),
            albu.OneOf([
                albu.MotionBlur(blur_limit=5),
                albu.MedianBlur(blur_limit=5),
                albu.GaussianBlur(blur_limit=5),
                albu.GaussNoise(var_limit=(5.0, 20.0)),
            ], p=0.5),
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return transforms


def get_test_augmentation(img_size):
    transforms = albu.Compose([
        albu.Resize(img_size, img_size, always_apply=True),
        albu.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transforms
