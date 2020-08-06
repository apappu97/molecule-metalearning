"""Trains a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import cross_validate
from chemprop.utils import create_logger
import wandb

if __name__ == '__main__':
    args = TrainArgs().parse_args()
    wandb.init(project='molecule-metalearning', name=args.experiment_name)
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    cross_validate(args, logger)
