"""Trains a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import meta_cross_validate
from chemprop.utils import create_logger, setup_wandb
import wandb
import time


if __name__ == '__main__':
    args = TrainArgs().parse_args()
    print("Setting args.meta_learning to True as we are meta learning")
    args.meta_learning = True
    wandb.init(name=args.experiment_name)
    setup_wandb(args)
    start_time = time.time()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    meta_cross_validate(args, logger)
    end_time = time.time()
    info = logger.info if logger is not None else print
    info('Total running time was {} seconds'.format(end_time - start_time))
    wandb.log({"total_time": end_time -  start_time})