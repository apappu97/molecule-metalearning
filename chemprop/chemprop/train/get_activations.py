import csv
from logging import Logger
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange, tqdm
import pickle
from torch import optim

from chemprop.args import TrainArgs
from chemprop.data import StandardScaler, MoleculeDataLoader
from chemprop.data import create_meta_data_loader
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import MoleculeModel, ANILMoleculeModel
from chemprop.nn_utils import param_count
from chemprop.utils import load_checkpoint,\
    makedirs, save_checkpoint, save_smiles_splits, create_logger
import wandb
import time

# @profile
def get_activations(args: TrainArgs, logger: Logger = None) -> List[float]:
    """
    Loads a pretrained meta learnt model, and for each test task, obtains activations and saves these as pickle files
    There is no randomness -- data loading must happen in the same order for everything.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    start_time = time.time()
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
    # set args.batch_size to be all the data at once, just to get all the activations at once 
    args.batch_size = len(data)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')
    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')


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
        logger=logger,
        shuffle_train= False,# don't shuffle the train dataset as we need the activations in the right order
        task_sim_flag = True) # we want to yield the entire train set in one batch

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
    def _load_model(args):
        if args.checkpoint_paths is not None:
            debug(f'Loading model')
            model = load_checkpoint(args.checkpoint_paths[0], logger=logger)
        else:
            debug(f'Building model from scratch')
            model = MoleculeModel(args)

        debug(f'Number of parameters = {param_count(model):,}')
        debug(model)
        if args.cuda:
            debug('Moving maml model to cuda')
        model = model.to(args.device)

        return model

    model = _load_model(args)
    # setup activations logging 
    activations_dict = {}
    
    def get_first_linear_layer_acts():
        def hook(model, input, output):
            activations_dict['ffn_1'] = output.T.detach()
        return hook
    
    def get_last_linear_layer_acts():
        def hook(model, input, output):
            sigmoid_func = nn.Sigmoid()
            sigmoid_acts = sigmoid_func(output.detach())
            activations_dict['ffn_2'] = sigmoid_acts.T.detach()
        return hook


    model.ffn[2].register_forward_hook(get_first_linear_layer_acts()) # registers on the activation 'layer' after the first linear layer
    model.ffn[-1].register_forward_hook(get_last_linear_layer_acts()) # registers hook on last linear layer and applies sigmoid before storing activations
    
    # Meta test -- evaluate with early stopping
    info('Beginning meta testing')

    task_counter = 0
    for meta_test_batch in tqdm(test_meta_task_data_loader, total=len(test_meta_task_data_loader)):
        for task in tqdm(meta_test_batch):
            batch_counter = 0
            for batch in task.train_data_loader:
                mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()
                # get gnn activations directly, i.e. the molecule vectors for each
                activations_dict['gnn'] = model.encoder(mol_batch, features_batch).T.detach()
                # the next call will call forward on the feed forward nets too and get those activations
                model(mol_batch, features_batch)
                torch.cuda.empty_cache()
                if batch_counter > 0:
                    print('This is bad, there should only be one batch, i.e. all train data in the task')
                batch_counter += 1
            if task_counter > 0:
                print('This is bad, there should only be one task utilized in this loop!')
            task_counter += 1

    info('Took {} seconds to complete getting activations in meta testing'.format(time.time() - start_time))
    
    # save appropriately
    save_file_name = args.results_save_dir + args.experiment_name + '.pickle'
    with open(save_file_name, 'wb') as handle:
        pickle.dump(activations_dict, handle)

if __name__ == '__main__':
    # parse args and set up task names to be just the one meta test task required typical of the meta cross validate code etc.
    args = TrainArgs().parse_args()
    args.meta_learning = True
    args.meta_test = True
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    get_activations(args, logger)