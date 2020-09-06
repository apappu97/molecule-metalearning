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
from chemprop.models import MoleculeModel, ANILMoleculeModel
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint, save_smiles_splits
import wandb
import time
import pdb
from memory_profiler import profile
from collections import deque

# @profile
def run_meta_testing(args: TrainArgs, logger: Logger = None) -> List[float]:
    """
    Loads a pretrained meta learnt model and tests on the meta test tasks

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

    assert args.chembl_assay_metadata_pickle_path is not None
    with open(args.chembl_assay_metadata_pickle_path + 'chembl_1024_assay_type_to_names.pickle', 'rb') as handle:
        chembl_1024_assay_type_to_names = pickle.load(handle)
    with open(args.chembl_assay_metadata_pickle_path + 'chembl_1024_assay_name_to_type.pickle', 'rb') as handle:
        chembl_1024_assay_name_to_type = pickle.load(handle)

    chembl_id_to_idx = {chembl_id: idx for idx, chembl_id in enumerate(args.task_names)}
    T_test = [0] * args.num_tasks
    test_task_id = chembl_id_to_idx[args.meta_test_task]
    T_test[test_task_id] = 1

    test_meta_task_data_loader = create_meta_data_loader(
        dataset=data,
        tasks=T_test,
        task_names=args.task_names,
        meta_batch_size=1, # so that we can yield one test task at a time during testing
        sizes=args.meta_test_split_sizes,
        cache=cache,
        args=args,
        logger=logger)

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
        if args.ANIL:
            maml_model = ANILMoleculeModel(model, fast_lr=args.inner_loop_lr)
        else:
            maml_model = l2l.algorithms.MAML(model, lr=args.inner_loop_lr, first_order=args.FO_MAML, allow_nograd=True)
        debug(f'Number of parameters = {param_count(maml_model):,}')
        debug(maml_model)
        if args.cuda:
            debug('Moving maml model to cuda')
        maml_model = maml_model.to(args.device)

        if type(maml_model) is l2l.algorithms.MAML:
            wandb.watch(maml_model.module, log='all')
        elif type(maml_model) is ANILMoleculeModel:
            wandb.watch(maml_model.molecule_model, log = 'all')
        else:
            raise ValueError("Wandb doesn't know how to watch this type of model")

        return maml_model

    maml_model = _setup_maml_model(args)
    # Ensure that model is saved in correct location for evaluation if 0 epochs
    if args.ANIL:
        maml_model_name = args.experiment_name + '_anil_model.pt'
    else:
        maml_model_name = args.experiment_name + '_maml_model.pt'

    # Meta test -- evaluate with early stopping
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