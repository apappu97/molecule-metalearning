import csv
from logging import Logger
import os
import sys
from typing import List

import numpy as np
import torch
from tqdm import trange
import pickle
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.args import TrainArgs
from chemprop.data import StandardScaler, MoleculeDataLoader
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint, save_smiles_splits

import wandb

def run_pretraining(args: TrainArgs, logger: Logger = None) -> List[float]:
    """
    Trains the pretraining baseline and saves the output model with early stopping

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Print command line
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')

    # Print args
    debug('Args')
    debug(args)

    # Save args
    args.save(os.path.join(args.save_dir, 'args.json'))

    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    # Get data
    debug('Loading data')
    args.task_names = args.target_columns or get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)
    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)
    
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Automatically determine whether to cache
    if len(data) <= args.cache_cutoff:
        cache = True
        num_workers = 0
    else:
        cache = False
        num_workers = args.num_workers

    # Create data loaders
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        cache=cache,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        cache=cache
    )

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = MoleculeModel(args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)
        wandb.watch(model, log='all')
        model_name = args.experiment_name + '.pt'
        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, model_name), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_scores = []
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )

            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            debug(f'Validation {args.metric} = {avg_val_score:.6f}')
            wandb.log({'validation_{}'.format(args.metric): avg_val_score})

            if args.show_individual_scores:
                # Individual validation scores
                individual_scores = {}
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    individual_scores['validation_{}_{}'.format(task_name, args.metric)] = val_score

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_scores, best_epoch = avg_val_score, val_scores, epoch
                save_checkpoint(os.path.join(save_dir, model_name), model, scaler, features_scaler, args)        

    return best_scores, best_epoch
