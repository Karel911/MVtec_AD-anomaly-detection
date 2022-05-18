import glob
import os
import warnings
import numpy as np
import pandas as pd
import torch

from trainer import Trainer, Tester
from util.utils import AvgMeter

warnings.filterwarnings('ignore')


class Wrapper():
    def __init__(self, args):
        self.args = args
        self.train_df = pd.read_csv('data/train/train_df.csv')
        self.loss = AvgMeter()
        self.f1 = AvgMeter()

        if args.train_method == 'one_class':
            self.label_unique = sorted(np.unique(self.train_df['class']))
        else:
            self.label_unique = self.unique_label(np.array(self.train_df['label']), one_class=False)

        self.loop_type = self.label_unique if self.args.train_method == 'one_class' else range(1, self.args.fold + 1)

    def train(self):
        train_method = '' if self.args.train_method == 'one_class' else 'Fold'

        for loop in self.loop_type:
            print(f'<---- {train_method}{loop} initializing ---->')

            if self.args.train_method == 'one_class':
                tr_gt, null_value = self.preprocess_label(unique_label=loop)
            else:
                tr_gt, null_value = self.preprocess_label(y=np.array(self.train_df['label']))

            save_path = os.path.join(self.args.model_path,
                                     f'{str(self.args.exp_num)}/{train_method}{loop}_E{self.args.arch}')

            os.makedirs(save_path, exist_ok=True)
            if self.args.train_method == 'one_class':
                val_loss, val_f1 = Trainer(self.args, save_path, unique_label=loop,
                                           tr_gt=tr_gt, null_value=null_value).init()
            else:
                val_loss, val_f1 = Trainer(self.args, save_path, fold=loop, tr_gt=tr_gt).init()

            self.loss.update(val_loss, n=1)
            self.f1.update(val_f1, n=1)

            print(f'{train_method}_{loop} mean : loss {self.loss.avg} | F1 score {self.f1.avg}')

    def test(self, submission, drop_lst):
        fold_features = torch.FloatTensor().cuda()
        test_method = '' if self.args.train_method == 'one_class' else 'Fold'

        for idx_loop, loop in enumerate(self.loop_type):
            if loop in drop_lst:
                pass
            else:
                save_path = os.path.join(self.args.model_path,
                                         f'{str(self.args.exp_num)}/{test_method}{loop}_E{self.args.arch}')
                best = [int(best.split('_')[-1].replace('.pth', '')) for best in glob.glob(save_path + '/model_*.pth')]
                weight_path = os.path.join(save_path, f'model_{max(best)}.pth')

                if self.args.train_method == 'one_class':
                    gt = self.train_df.label.apply(lambda x: x if x.split('-')[0] == loop else 'null')
                    _, unique = self.unique_label(gt, one_class=True)
                    num_classes = len(unique)

                    print(f'<---- {loop} initializing ---->')
                    features = Tester(self.args, weight_path, num_classes=num_classes).test()

                    self.make_submission(features, unique, submission)

                    self.check_sub()
                    print('Submission file created')

                else:
                    print(f'<---- {test_method}-{loop} initializing ---->')
                    features = Tester(self.args, weight_path, num_classes=88).test()

                    fold_features = torch.cat([fold_features, features[None, :, :]], dim=0)

                    if self.args.ensemble and idx_loop == (loop - 1):
                        self.make_submission(fold_features, self.label_unique, submission)
                        print('Submission file created')

    def preprocess_label(self, y=None, unique_label=None):
        if self.args.train_method == 'one_class':
            tr_gt = self.train_df.label.apply(lambda x: x if x.split('-')[0] == unique_label else 'null')
            null_value, unique = self.unique_label(tr_gt, one_class=True)
        else:
            tr_gt = y
            null_value = None
            unique = self.unique_label(tr_gt, one_class=False)

        tr_gt = np.array([unique[k] for k in tr_gt])

        return tr_gt, null_value

    def unique_label(self, gt, one_class=True):
        unique = sorted(np.unique(gt))
        unique = {key: value for key, value in zip(unique, range(len(unique)))}

        if one_class:
            null_value = unique['null']
            return null_value, unique
        else:
            return unique

    def make_output(self, features, unique):
        if self.args.train_method != 'one_class':
            features = features.mean(0)

        preds = features.argmax(1).detach().cpu().numpy().tolist()
        label_decoder = {val: key for key, val in unique.items()}
        output = [label_decoder[result] for result in preds]

        return output

    def check_sub(self):
        submission = pd.read_csv(f'test_{self.args.train_method}_{self.args.exp_num}.csv')
        cols = np.array(submission.iloc[:, 2:], dtype=str)
        check = np.zeros(cols.shape[0])

        for i in range(cols.shape[1]):
            idx = cols[:, i] != 'null'
            pred = cols[idx, i]
            submission.label.iloc[idx] = pred
            check[idx] += 1
            print('checking the data integrity : ', check[check > 1])

        submission.to_csv(f"sub_{self.args.train_method}_{self.args.exp_num}.csv", index=False)

    def make_submission(self, features, unique, submission):
        sub_str = 'test' if self.args.train_method == 'one_class' else 'sub'

        sub = self.make_output(features, unique)

        submission['label'] = sub
        submission.to_csv(f"{sub_str}_{self.args.train_method}_{self.args.exp_num}.csv", index=False)
