from logging import Logger
import os
from typing import Tuple

import numpy as np

from .run_meta_training import run_meta_training
from .run_meta_testing import run_meta_testing
from chemprop.args import TrainArgs
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs, save_results
import pdb
import pickle

def meta_cross_validate(args: TrainArgs, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    if args.seeds and (len(args.seeds) != args.num_folds):
        raise ValueError("Length of args seeds and num folds must be equal if seeds is provided")
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = args.target_columns or get_task_names(args.data_path)
    
    # Run training on different random seeds for each fold
    all_scores = []
    all_best_epochs = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        if args.seeds:
            args.seed = args.seeds[fold_num]
        else:
            args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        if args.meta_test:
            model_scores, meta_best_epochs = run_meta_testing(args, logger)
        else:
            model_scores, meta_best_epochs = run_meta_training(args, logger)
        all_scores.append(model_scores)
        all_best_epochs.append(meta_best_epochs)
    all_scores = np.array(all_scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    # Save results for later analysis
    if args.meta_test:
        test_task_names = [args.meta_test_task]
    else:
        with open(args.chembl_assay_metadata_pickle_path + 'chembl_1024_meta_test_task_split.pickle', 'rb') as handle:
            T_test = pickle.load(handle)
        test_task_names = [task_names[idx] for idx in np.nonzero(T_test)[0]]
    save_results(all_scores, all_best_epochs, test_task_names, args)

    if args.show_individual_scores:
        for task_num, task_name in enumerate(test_task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score
