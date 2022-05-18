import torch
import numpy as np
import pandas as pd


def merge_fold(feat, fold_list):
    min = 99
    agg_sum = None

    for i in range(feat.shape[0]):
        if i in fold_list:
            if i < min:
                min = i
                agg_sum = feat[i, :, :]
                agg_sum = agg_sum[None, :, :]
            else:
                agg_sum = torch.cat([agg_sum, feat[i, :, :].unsqueeze(0)], dim=0)

    return agg_sum


def drop_fold(feat, fold_list):
    min = 99
    agg_sum = None

    for i in range(feat.shape[0]):
        if i in fold_list:
            pass
        else:
            if i < min:
                min = i
                agg_sum = feat[i, :, :]
                agg_sum = agg_sum[None, :, :]
            else:
                agg_sum = torch.cat([agg_sum, feat[i, :, :].unsqueeze(0)], dim=0)

    return agg_sum


def decode_label(features):
    tr_gt = np.array(pd.read_csv('data/train/train_df.csv')['label'])
    label_unique = sorted(np.unique(tr_gt))
    label_unique = {key: value for key, value in zip(label_unique, range(len(label_unique)))}
    label_decoder = {val: key for key, val in label_unique.items()}

    features = features.argmax(1).tolist()
    new = [label_decoder[result] for result in features]
    return new


# load data
tr_gt = np.array(pd.read_csv('data/train/train_df.csv')['label'])
label_unique = sorted(np.unique(tr_gt))
label_unique = {key: value for key, value in zip(label_unique, range(len(label_unique)))}

baseline2 = torch.load('results/features_146.pth')
feat39 = torch.load('results/features_39(640).pth')
feat66 = torch.load('results/features_66(640).pth')
feat85 = torch.load('results/features_85(640).pth')
feat92 = torch.load('results/features_92(576).pth')


'''
Baseline1 (LB 0.896): Efficient-B6 (66) & Efficient-B6 (85, ugmentation including rotation of 45 degree)

soft_ensemble (LB 0.899): Efficient-B6 (66) & Efficient-B6 (85, rotate 45) & 
            Efficient-B6 (39, arcFace loss) & Efficient-B6 (92, arcFace & label smoothing loss)
            
Baseline2 (LB 0.894): Efficient-B7 (146, label smoothing loss)
'''

# baseline1
feat_66_85 = torch.cat([drop_fold(feat66, [0, 1, 5, 6]), merge_fold(feat85, [0, 1])]).mean(0)
baseline1 = decode_label(feat_66_85)

# baseline2
baseline2 = decode_label(baseline2.mean(0))

# baseline3
soft_ensemble = torch.cat([drop_fold(feat66, [0, 1, 5, 6]), merge_fold(feat85, [0, 1])]).mean(0) * .6\
      + merge_fold(feat39, [2]).mean(0) * .1 + merge_fold(feat39, [8]).mean(0) * .1 + \
      merge_fold(feat92, [0]).mean(0) * .15
soft_ensemble = decode_label(soft_ensemble)


"""
HARD ENSEMBLE WITH HYPOTHESIS 1

Observation 1-1: On the overall validation results, the anomaly class-combined was relatively hard to predict.
Observation 1-2: Baseline1 has relatively better performance for the normal and anomaly-combined on the validation results.

Hypothesis 1: Trust the results from the baseline1 and blend the new perspective from the soft_ensemble result 
              except for both normal and anomaly-combined.

Result: LB 0.902    
          
Priority 1: Trust a new perspective for a anomaly class except for the combined class.
Priority 2: Trust normal class and anomaly class-combined from the baseline1.
"""

ensemble = baseline1

for i, (base, agg) in enumerate(zip(baseline1, soft_ensemble)):
    if (base != agg) and (agg.split('-')[-1] != 'good'):
        if base.split('-')[-1] == 'combined':
            pass
        else:
            ensemble[i] = agg


"""
HARD ENSEMBLE WITH HYPOTHESIS 2

Observation 2-1: The baseline2 showed better performance for the classes tile, carpet, and zipper on the validation 
                 compared to the baseline1.
Observation 2-2: The baseline2 exhibited a different view for the anomaly class-combined on the validation
                 compared to the baseline1.
              
Hypothesis 2-1: Trust the new perspective from the baseline2 results including the classes tile, carpet, and zipper.
Hypothesis 2-2: Trust the new perspective for the anomaly class-combined from the baseline2 results.


Result: LB 0.9099

Priority 1: Trust a new perspective for the anomaly class-combined (modified).
Priority 2: Trust a new perspective for anomaly classes tile, carpet, and zipper from the baseline2.
Priority 3: Trust normal and anomaly class-combined from the baseline1.
"""

for i, (existing, new) in enumerate(zip(ensemble, baseline2)):
    if existing != new:
        if (new.split('-')[0] == 'tile') or (new.split('-')[0] == 'carpet') \
                or (new.split('-')[0] == 'zipper') or (new.split('-')[-1] == 'combined'):
            ensemble[i] = new


"""
HARD ENSEMBLE WITH HYPOTHESIS 3

Hypothesis 3: One-class based self supervised learning model has a different view compared with the baseline1 and 2.

The combination of labels, 'cable', 'grid', 'metal_nut', 'pill', and 'wood', showed the best performance.

Result: LB-Private 0.926

Priority 1: Trust a new perspective for the anomaly class-combined.
Priority 2: Trust a new perspective for anomaly classes tile, carpet, and zipper from the baseline1.
Priority 3: Trust a new perspective for anomaly classes cable, grid, metal_nut, pill, and wood 
            from the one-class based self supervised learning model.
Priority 4: Trust normal class from the baseline1 (modified).
"""

label = ['cable', 'grid', 'metal_nut', 'pill', 'wood']

for target in label:
    df = pd.read_csv(f'results/validation_E6_87_{target}.csv').fillna(0)
    df[f'{target}'] = 'none'

    arr = np.array(df, dtype=str)
    for i in range(arr.shape[0]):
        if arr[i, 0] == 0:
            pass
        else:
            u, n = np.unique(arr[i, :], return_counts=True)
            df[f'{target}'][i] = u[n == max(n)]

    for i, (existing, new) in enumerate(zip(list(ensemble), df[f'{target}'])):
        if (existing.split('-')[0] == target) and (existing != new[0]) and (new[0].split('-')[1] != 'good'):
            ensemble[i] = new[0]

submission = pd.read_csv("data/sample_submission.csv")
submission.label = ensemble
submission.to_csv('1st_place_reproduce_results.csv', index=False)