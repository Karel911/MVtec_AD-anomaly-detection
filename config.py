import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='train', help='Model Training or Testing options')
    parser.add_argument('--exp_num', default=0, type=str, help='experiment_number')
    parser.add_argument('--data_path', type=str, default='data/')

    # Model parameter settings
    parser.add_argument('--arch', type=str, default='4', help='Backbone Architecture')
    parser.add_argument('--margin', type=float, default=0.1, help='additive angular margin')
    parser.add_argument('--scale', type=int, default=8, help='normalized cos_theta re-scaler')
    parser.add_argument('--model_name', type=str, default='resnest50d_4s2x40d', help='vit model name')
    parser.add_argument('--model', type=str, default='efficientnet', help='model use')

    # Training parameter settings
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--criterion', type=str, default='smoothing', help='ce or smoothing')
    parser.add_argument('--scheduler', type=str, default='Reduce', help='Reduce or Step')
    parser.add_argument('--aug_ver', type=int, default=2, help='1=Normal, 2=Hard')
    parser.add_argument('--lr_factor', type=float, default=0.1)
    parser.add_argument('--clipping', type=float, default=2, help='Gradient clipping')
    parser.add_argument('--patience', type=int, default=5, help="Scheduler ReduceLROnPlateau's parameter & Early Stopping(+5)")
    parser.add_argument('--model_path', type=str, default='results/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=10)
    parser.add_argument('--ensemble', type=bool, default=True)
    parser.add_argument('--train_method', type=str, default='baseline', help='baseline or one_class')
    parser.add_argument('--arcloss', type=str, default='false', help='applying arcFace loss')


    # Hardware settings
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    cfg = parser.parse_args()

    return cfg


if __name__ == '__main__':
    cfg = getConfig()
    cfg = vars(cfg)
    print(cfg)