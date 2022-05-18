import pprint
import random
from config import getConfig
from wrapper import *
warnings.filterwarnings('ignore')
args = getConfig()


def main(args):
    print('<---- Training Params ---->')
    pprint.pprint(args)

    # Random Seed
    np.seterr(all="ignore")
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.action == 'train':
        Wrapper(args).train()

    else:
        submission = pd.read_csv("data/sample_submission.csv")

        if args.train_method == 'one_class':
            drop_class = []
            Wrapper(args).test(submission, drop_class)
        else:
            drop_fold = []
            Wrapper(args).test(submission, drop_fold)


if __name__ == '__main__':
    main(args)