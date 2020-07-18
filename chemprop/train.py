"""Trains a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import cross_validate
from chemprop.utils import create_logger, setup_wandb
import time
import wandb

if __name__ == '__main__':

    args = TrainArgs().parse_args()
    wandb.init(name=args.experiment_name)
    setup_wandb(args)
    start_time = time.time()
    wandb.log({"start_time": start_time})
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    cross_validate(args, logger)
    end_time = time.time()
    info = logger.info if logger is not None else print
    info('Total running time was {} seconds'.format(end_time - start_time))
    wandb.log({"end_time": end_time, "elapsed_time": end_time -  start_time})
