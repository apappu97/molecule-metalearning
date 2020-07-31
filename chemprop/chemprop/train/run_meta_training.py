import csv
from logging import Logger
import os
import sys
from typing import List

import numpy as np
import torch
from tqdm import trange
import pickle
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
import learn2learn as l2l

from .meta_evaluate import meta_evaluate, meta_test
from .predict import predict
from .meta_train import meta_train
from chemprop.args import TrainArgs
from chemprop.data import StandardScaler, MoleculeDataLoader, create_meta_data_loader
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint, save_smiles_splits
import wandb
import time
import pdb
from memory_profiler import profile
from collections import deque

# @profile
def run_meta_training(args: TrainArgs, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

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

    # TODO -- should this change every fold, i.e. by using args.seed? args.pytorch_seed never changes! So different folds use the same weight initialisation.
    # Set pytorch and numpy seeds for random initial weights
    # torch.manual_seed(args.pytorch_seed)
    # TODO -- for now, using the args seed which *also* changes on each fold, leading to a new torch initialisation of the architecture.
    # For initial experiments this is no big deal, as we only run with num folds 1, so it's effectively the same as the above line.
    torch.manual_seed(args.seed)

    # Get data
    debug('Loading data')
    args.task_names = args.target_columns or get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')
    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

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

    # Set up MetaTaskDataLoaders, which takes care of task splits under the hood 
    # Set up task splits into T_tr, T_val, T_test

    """ 
    Load ChEMBL task splits. Same in spirit as GSK implementation of task splits.
    We have 5 Task types remaining
    ADME (A)
    Toxicity (T)
    Unassigned (U) 
    Binding (B)
    Functional (F)
    resulting in 902 tasks.
    """

    assert args.chembl_assay_metadata_pickle_path is not None
    with open(args.chembl_assay_metadata_pickle_path + 'chembl_1024_assay_type_to_names.pickle', 'rb') as handle:
        chembl_1024_assay_type_to_names = pickle.load(handle)
    with open(args.chembl_assay_metadata_pickle_path + 'chembl_1024_assay_name_to_type.pickle', 'rb') as handle:
        chembl_1024_assay_name_to_type = pickle.load(handle)

    chembl_id_to_idx = {chembl_id: idx for idx, chembl_id in enumerate(args.task_names)}
    with open(args.chembl_assay_metadata_pickle_path + 'chembl_1024_meta_train_task_split.pickle', 'rb') as handle:
        T_tr = pickle.load(handle)
    with open(args.chembl_assay_metadata_pickle_path + 'chembl_1024_meta_val_task_split.pickle', 'rb') as handle:
        T_val = pickle.load(handle)
    with open(args.chembl_assay_metadata_pickle_path + 'chembl_1024_meta_test_task_split.pickle', 'rb') as handle:
        T_test = pickle.load(handle)

    if args.dummy:
        """
        Random task split for testing of *REDUCED* size, hence the 0.005 splits
        """
        print("Running in dummy mode")
        task_indices = list(range(len(args.task_names)))
        # np.random.shuffle(task_indices)
        train_task_split, val_task_split, test_task_split = 0.005, 1, 1 # just use a fraction of the train tasks, but all val and test tasks
        # train_task_cutoff = int(len(T_tr) * train_task_split)
        # val_task_cutoff = train_task_cutoff + int(len(task_indices)*val_task_split)
        # test_task_cutoff = val_task_cutoff + int(len(task_indices) * test_task_split)

        actual_tr_tasks = np.nonzero(T_tr)[0]
        actual_val_tasks = np.nonzero(T_val)[0]
        actual_test_tasks = np.nonzero(T_test)[0]

        T_tr, T_val, T_test = [0] * len(task_indices), [0] * len(task_indices), [0] * len(task_indices)
        for idx in range(0, int(train_task_split*len(actual_tr_tasks))):
            T_tr[idx] = 1
        for idx in range(int(val_task_split * len(actual_val_tasks))):
            T_val[idx] = 1
        for idx in range(int(test_task_split * len(actual_test_tasks))):
            T_test[idx] = 1

        # for idx in task_indices[:train_task_cutoff]:
        #     T_tr[idx] = 1
        # for idx in task_indices[train_task_cutoff:val_task_cutoff]:
        #     T_val[idx] = 1
        # for idx in task_indices[val_task_cutoff:test_task_cutoff]:
        #     T_test[idx] = 1

    train_meta_task_data_loader = create_meta_data_loader(
        dataset=data,
        tasks=T_tr,
        task_names=args.task_names,
        meta_batch_size=args.meta_batch_size,
        sizes=args.meta_train_split_sizes,
        args=args,
        logger=logger)
    # train_meta_task_data_loader = MetaTaskDataLoader(
    #         dataset=data,
    #         tasks=T_tr,
    #         task_names=args.task_names,
    #         meta_batch_size=args.meta_batch_size,
    #         num_workers=args.num_workers,
    #         sizes=args.meta_train_split_sizes,
    #         args=args,
    #         logger=logger)

    val_meta_task_data_loader = create_meta_data_loader(
        dataset=data,
        tasks=T_val,
        task_names=args.task_names,
        meta_batch_size=args.meta_batch_size,
        sizes=args.meta_train_split_sizes,
        args=args,
        logger=logger)

    # val_meta_task_data_loader = MetaTaskDataLoader(
    #         dataset=data,
    #         tasks=T_val,
    #         task_names=args.task_names,
    #         meta_batch_size=args.meta_batch_size,
    #         num_workers=args.num_workers,
    #         sizes=args.meta_train_split_sizes,
    #         args=args,
    #         logger=logger)

    test_meta_task_data_loader = create_meta_data_loader(
        dataset=data,
        tasks=T_test,
        task_names=args.task_names,
        meta_batch_size=1, # so that we can yield one test task at a time during testing
        sizes=args.meta_test_split_sizes,
        args=args,
        logger=logger)

    # test_meta_task_data_loader = MetaTaskDataLoader(
    #         dataset=data,
    #         tasks=T_test,
    #         task_names=args.task_names,
    #         meta_batch_size=1,
    #         num_workers=args.num_workers,
    #         sizes=args.meta_test_split_sizes,
    #         args=args,
    #         logger=logger)

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        raise ValueError("This script can only be run on ChEMBL classification")
    else:
        scaler = None

    # Set up save dir
    save_dir = os.path.join(args.save_dir, f'maml_model')
    makedirs(save_dir)

    # Load/build model
    # Only set up for one model, no ensembling
    def _setup_maml_model(args):
        if args.checkpoint_paths is not None:
            debug(f'Loading model')
            model = load_checkpoint(args.checkpoint_paths[0], logger=logger)
        else:
            debug(f'Building model ')
            model = MoleculeModel(args)

        # Keep it simple, we are using fixed outer and inner loop LRs and ADAM optimizer. 
        maml_model = l2l.algorithms.MAML(model, lr=args.inner_loop_lr, first_order=args.FO_MAML, allow_nograd=True)
        debug(f'Number of parameters = {param_count(maml_model):,}')
        debug(maml_model)
        if args.cuda:
            debug('Moving maml model to cuda')
        maml_model = maml_model.to(args.device)

        if type(maml_model) is l2l.algorithms.MAML:
            wandb.watch(maml_model.module, log='all')
        else:
            raise ValueError("Wandb doesn't know how to watch this type of model")

        return maml_model

    maml_model = _setup_maml_model(args)
    # Ensure that model is saved in correct location for evaluation if 0 epochs
    maml_model_name = args.experiment_name + '_maml_model.pt'
    save_checkpoint(os.path.join(save_dir, maml_model_name), maml_model, scaler=scaler, features_scaler=None, args=args)

    # Optimizers
    # optimizer = build_optimizer(model, args)

    # Learning rate schedulers
    # scheduler = build_lr_scheduler(optimizer, args)

    meta_opt = optim.Adam(maml_model.parameters(), args.outer_loop_lr)
    # Run training
    best_score = float('inf') if args.minimize_score else -float('inf')
    best_epoch = 0 
    # Initializing the loss queue 
    loss_queue = deque() 
    for epoch in trange(args.epochs):
        debug(f'Epoch {epoch}')
        start_time = time.time()
        meta_train_loss = meta_train(
            maml_model=maml_model,
            meta_task_data_loader=train_meta_task_data_loader,
            epoch=epoch,
            loss_func=loss_func,
            loss_queue=loss_queue,
            meta_optimizer=meta_opt,
            args=args,
            logger=logger
        )
        info('Took {} seconds to complete one epoch of meta training'.format(time.time() - start_time))
        # No annealing / stepping as we are using a fixed learning rate for inner and outer loop
        # if isinstance(scheduler, ExponentialLR):
        #     scheduler.step()

        # meta validation to determine whether to save a new checkpoint 
        val_task_scores, meta_val_loss = meta_evaluate(
            maml_model=maml_model,
            meta_task_data_loader=val_meta_task_data_loader,
            num_inner_gradient_steps=args.num_inner_gradient_steps,
            metric_func=metric_func,
            loss_func=loss_func,
            dataset_type=args.dataset_type,
            logger=logger
        )
        info('Took {} seconds to complete one epoch of meta training and validating'.format(time.time() - start_time))

        # Average validation score
        avg_val_score = np.nanmean(val_task_scores)
        debug(f'Meta validation score {args.metric} = {avg_val_score:.6f}')
        wandb.log({'meta_train_epoch_loss': meta_train_loss, 'meta_val_epoch_loss': meta_val_loss, 'meta_val_score': avg_val_score, 'epoch': epoch})

        if args.show_individual_scores:
            # Individual validation scores
            for task_name, val_score in zip(val_meta_task_data_loader.meta_task_names, val_task_scores):
                debug(f'Meta validation {task_name} {args.metric} = {val_score:.6f}')
                wandb.log({'Meta validation {} {}'.format(task_name, args.metric): val_score})

        # Save model checkpoint if improved validation score
        if args.minimize_score and avg_val_score < best_score or \
                not args.minimize_score and avg_val_score > best_score:
            best_score, best_epoch = avg_val_score, epoch
            info('Found better MAML checkpoint after meta validation, saving now')
            save_checkpoint(os.path.join(save_dir, maml_model_name), maml_model, scaler=scaler, args=args)        
            wandb.save(os.path.join(save_dir, maml_model_name))
    
    # Evaluate on test set using model with best validation score
    info(f'Best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')

    def _load_maml_model(save_dir, maml_model_name, args):
        model = load_checkpoint(os.path.join(save_dir, maml_model_name), device=args.device, logger=logger)
        maml_model = l2l.algorithms.MAML(model, lr=args.meta_test_lr, first_order=args.FO_MAML, allow_nograd=True)
        return maml_model
    
    maml_model = _load_maml_model(save_dir, maml_model_name, args)
    # Meta test time -- evaluate with early stopping
    info('Beginning meta testing')
    start_time = time.time()
    test_scores, best_epochs = meta_test(
            maml_model, 
            test_meta_task_data_loader, 
            metric_func=metric_func, 
            loss_func=loss_func,
            dataset_type=args.dataset_type, 
            meta_test_epochs = args.meta_test_epochs,
            save_dir=save_dir,
            args=args,
            logger=logger)
    info('Took {} seconds to complete meta testing'.format(time.time() - start_time))
    # pdb.set_trace()
    # Average test score
    avg_test_score = np.nanmean(test_scores)
    info(f'Model test {args.metric} = {avg_test_score:.6f}')

    return test_scores, best_epochs