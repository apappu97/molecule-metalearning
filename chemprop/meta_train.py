"""Trains a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import meta_cross_validate
from chemprop.utils import create_logger


if __name__ == '__main__':
    args = TrainArgs().parse_args()
    print("Setting args.meta_learning to True as we are meta learning")
    args.meta_learning = True
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    meta_cross_validate(args, logger)
