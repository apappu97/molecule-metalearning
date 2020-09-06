"""Tests a meta trained model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import meta_cross_validate
from chemprop.utils import create_logger, setup_wandb
import wandb
import time

if __name__ == '__main__':
    args = TrainArgs().parse_args()
    if not args.meta_test_task:
        raise ValueError("Running meta test, expecting a meta test task")
    if args.checkpoint_paths is None:
        raise ValueError("Checkpoint file must be provided to run meta testing")

    print("Setting args.meta_learning to True as we are meta learning and setting args.meta_test to True as we are meta testing")
    args.meta_learning = True
    args.meta_test = True
    wandb.init(project='molecule-metalearning', name=args.experiment_name)
    setup_wandb(args)
    start_time = time.time()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    meta_cross_validate(args, logger)
    end_time = time.time()
    info = logger.info if logger is not None else print
    info('Total running time was {} seconds'.format(end_time - start_time))
    wandb.log({"total_time": end_time -  start_time})