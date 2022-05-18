# MVtec AD dataset anomaly detection (Image Classification)

The following codes are the solutions (1st place, private score: 0.92708) for the dacon competition.  
If you would like to know more about the competition, please refer to the following link:  
https://dacon.io/competitions/official/235894/overview/description

Briefly, the task is to classify an extremely imbalanced images into 88 classes 
in which the label is composed of class-state pairs.  
To solve this problem, we used Efficientnet and Resnext as a backbone with different training methods (e.g. one-class self-supervised learning, arcFace loss, and  label smoothing).  
We also blended the model weights and prediction results based on the validation results and hypothesis.  
Please refer to [ensemble.py](https://github.com/Karel911/MVtec_AD-anomaly-detection/blob/main/ensemble.py) to see how we blended the model weights and prediction results.  

## Used Models
The models we used to make the best prediction are as follows:
* features_66: EfficientNet-B6, 10-Fold, baseline, inference img_size=640, inference batch_size=64
* features_85: EfficientNet-B6, 10-Fold, baseline, added augmentation (rotate 45 degree), inference img_size=640, inference batch_size=64
* features_146: EfficientNet-B6, 10-Fold, baseline, label smoothing applied


* features_39: EfficientNet-B6, 10-Fold, arcFace loss applied, inference img_size=640, inference batch_size=64
* features_92: EfficientNet-B6, 5-Fold, label smoothing with arcFace applied, inference img_size=640, inference batch_size=64
* features_100: ResNext, 5-Fold, resnext101_32x8d, batch_size=16, fold=5
* validation_E6_87.csv: EfficientNet-B6, one-class self-supervised learning strategy, arcFace loss applied


## Data Directory
You can download the dataset at here: https://dacon.io/competitions/official/235894/data
<pre><code>
anomaly
├── data
│   ├── 5-Fold_idx.npy
│   ├── 10-Fold_idx.npy
│   ├── train
│   │   ├── images
│   │   ├── train_df.csv
│   ├── test
│   │   ├── images
│   ├── sample_submission.csv
      .
      .
      .
</code></pre>

## Requirements
* albumentations >= 1.1.0
* opencv-python >=4.5.5.64
* pandas >= 1.3.5
* scikit-learn >= 1.0.2
* timm >= 0.5.4
* torch >= 1.8.2
* torchvision >= 0.9.2
* tqdm >=4.64.0

## Run
* To run the codes, use the following commands:<br>
<pre><code>
# EfficientNet-B6 with label smoothing
python main.py train --arch 6 --img_size 576 --criterion smoothing

# Different backbone model (example of ResNest)
python main.py train --model resnest --model_name resnest50d_4s2x40d

# One-class training with ArcFace
python main.py train --train_method one_class --arcloss arcface

</code></pre>
* You can choose either 10- or 5-fold to train the model.  
* We used the fixed training and validation index for each fold for solid experiment.

## Configurations
* arch: EfficientNet backbone scale (E0 ~ E7)
* model: backbone model (e.g. efficientnet and resnest)
* model_name: name of the pretrained model that is available in timm
* train_method: training strategy 
* arcloss: ArcFace usage (default: False)
* fold: number of folds (different number of folds other than 5 or 10 is not available due to the fixed index)
* multi_gpu: multi-gpu learning options




