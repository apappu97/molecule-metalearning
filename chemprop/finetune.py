"""Trains a model on a dataset."""

from chemprop.args import TrainArgs
from chemprop.train import finetune_cross_validate
from chemprop.utils import create_logger
import wandb

if __name__ == '__main__':
    args = TrainArgs().parse_args()
    wandb.init(project='molecule-metalearning', name= 'finetuning_' + args.experiment_name)
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    finetune_cross_validate(args, logger)
