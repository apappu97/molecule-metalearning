"""Trains a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import pretraining_cross_validate
from chemprop.utils import create_logger
import wandb

if __name__ == '__main__':
    args = TrainArgs().parse_args()
    wandb.init(project='molecule-metalearning', name='pretraining_' + args.experiment_name)
    print('Setting args split sizes to 0.8 0.2 0 as this is pretraining, no test set needed')
    args.split_sizes = (0.8, 0.2, 0)
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    pretraining_cross_validate(args, logger)