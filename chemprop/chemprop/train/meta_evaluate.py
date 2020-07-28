import os
import logging
from typing import Callable, List
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import numpy as np

from .predict import predict
from .evaluate import evaluate_predictions
from chemprop.data import MoleculeDataLoader, StandardScaler
from chemprop.data import MetaTaskDataLoader, TaskDataLoader
from .meta_train import fast_adapt, predict_on_batch_and_return_loss, get_task_idx
from chemprop.utils import save_checkpoint, load_checkpoint
import wandb
from memory_profiler import profile
import pdb

def _eval_trained_model(learner, task_dataloader, targets, metric_func, dataset_type, logger):
    """
    Takes a trained model and just evaluates predictions on a given dataloader

    learner: fast adapted model
    task_dataloader: dataloader for desired split to evaluate learner on
    targets: targets for this dataloader 
    metric_func: metric func
    dataset_type: dataset type
    logger: logger
    """

    # Evaluate predictions now 
    learner.eval()
    preds = predict(
        model=learner,
        data_loader=task_dataloader,
        scaler=None
    )
    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        num_tasks=1,
        metric_func=metric_func,
        dataset_type=dataset_type,
        logger=logger
    )

    if len(results) == 1:
        results = results[0]
    return results

def _meta_eval_on_task(maml_model, task, loss_func, metric_func, num_inner_gradient_steps, dataset_type, logger):
    """
    Fast adapt and evaluate on a single task, used during meta evaluation

    maml_model: The MAML wrapped model
    task: TaskDataLoader object
    loss_func: loss function
    metric_func: metric function
    dataset_type: dataset type
    logger: logger
    """
    learner = maml_model.clone()
    curr_task_target_idx = get_task_idx(task)
    fast_adapt(learner, task, curr_task_target_idx, loss_func, num_inner_gradient_steps)
    results = _eval_trained_model(learner, task.val_data_loader, task.get_targets('val'), metric_func, dataset_type, logger)
    
    return results

# @profile
def meta_evaluate(maml_model,
             meta_task_data_loader: MetaTaskDataLoader,
             num_inner_gradient_steps: int,
             metric_func: Callable,
             loss_func: Callable,
             dataset_type: str,
             logger: logging.Logger = None) -> List[float]:
    """
    Evaluates a MAML model on a set of meta validation tasks. 

    For each task, fast adapts, records the metric, and averages the metric over all validation tasks.

    Returns average validation score.

    :param maml_model: An l2l wrapped model.
    :param data_loader: A MoleculeDataLoader.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param dataset_type: Dataset type.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """

    val_task_results = []
    for meta_val_batch in tqdm(meta_task_data_loader.tasks(), total = len(meta_task_data_loader)):
        meta_task_losses = 0.0
        for task in tqdm(meta_val_batch):
            results = _meta_eval_on_task(maml_model, task, loss_func, metric_func, num_inner_gradient_steps, dataset_type, logger)
            val_task_results.append(results)

    return val_task_results

def _train_epoch(learner, task, curr_task_target_idx, loss_func):
    learner.train()
    task_adaptation_loss = 0.0
    for batch in task.train_data_loader:
        adaptation_loss = predict_on_batch_and_return_loss(learner, batch, curr_task_target_idx, loss_func)
        # first order=True prevents the computational graph from being retained, c.f. https://github.com/learnables/learn2learn/issues/154
        learner.adapt(adaptation_loss, first_order=True) 
        batch_avg_loss = adaptation_loss.item()
        wandb.log({'meta_test_{}_adaptation_loss'.format(task.assay_name): batch_avg_loss})
        task_adaptation_loss += batch_avg_loss
    
    del adaptation_loss
    task_adaptation_loss /= len(task.train_data_loader) # normalize by number of batches to get avg batch mean loss 
    return task_adaptation_loss

def _meta_test_on_task(maml_model, task, meta_test_epochs, loss_func, metric_func, dataset_type, args, save_dir, logger):
    """
    Performs meta testing on a SINGLE task, i.e. training with early stopping and then final testing 

    maml_model: MAML wrapped model 
    task: TaskDataLoader
    meta_test_epochs: Num epochs to train 
    loss_func: loss function
    metric_func: metric function for task
    save_dir: dir to save checkpoints
    logger: logger
    """

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print
    info("Meta testing on task: {}".format(task.assay_name))    
    learner = maml_model.clone()
    curr_task_target_idx = get_task_idx(task)

    best_val_loss = float('inf')
    best_epoch = 0
    # save the best performing model in case it occurred at 0 epochs 
    save_checkpoint(os.path.join(save_dir, 'meta_test_{}_model.pt'.format(task.assay_name)), learner, scaler=None, features_scaler=None, args=args)

    for epoch in trange(meta_test_epochs):
        # train model for one epoch
        task_adaptation_loss = _train_epoch(learner, task, curr_task_target_idx, loss_func)
        # wandb.log({'meta_test_{}_epoch_adaptation_loss'.format(task.assay_name): task_adaptation_loss})
        
        # validate model  
        task_val_loss = 0.0
        learner.eval()
        with torch.no_grad():
            for batch in task.val_data_loader:
                loss = predict_on_batch_and_return_loss(learner, batch, curr_task_target_idx, loss_func)
                batch_avg_loss = loss.item()
                task_val_loss += batch_avg_loss
        
        task_val_loss /= len(task.val_data_loader) # normalize by number of batches to get avg batch loss
        wandb.log({'meta_test_{}_epoch_val_loss'.format(task.assay_name): task_val_loss})

        if task_val_loss < best_val_loss:
            info('New best model for test task {} at epoch {} with val loss {}'.format(task.assay_name, epoch + 1, task_val_loss))
            best_val_loss = task_val_loss
            best_epoch = epoch
            # save best checkpoint at meta test time 
            save_checkpoint(os.path.join(save_dir, 'meta_test_{}_model.pt'.format(task.assay_name)), learner, args=args)  
        else:
            info('Val loss: {}'.format(task_val_loss))   
    info('Finished early stopping for task {}, beginning testing'.format(task.assay_name))
    
    # Now that early stopping has identified the best model, calculate test loss
    model = load_checkpoint(os.path.join(save_dir, 'meta_test_{}_model.pt'.format(task.assay_name)))
    results = _eval_trained_model(model, task.test_data_loader, task.get_targets('test'), metric_func, dataset_type, logger)
    # pdb.set_trace() # look at NVIDIA memory usage here 
    return results, best_epoch

def meta_test(maml_model,
            meta_task_data_loader: MetaTaskDataLoader,
            metric_func: Callable,
            loss_func: Callable,
            dataset_type: str,
            meta_test_epochs: int,
            save_dir: str,
            args,
            logger: logging.Logger = None,
            fixed_inner_grad_steps: bool = False,
            num_inner_gradient_steps: int = 0) -> List[str]:
    """
    Evaluates model at meta test time.
    Main difference is that the maml_model is fast adapted with early stopping as opposed to for a fixed number of iterations
    """ 
    test_task_results = []
    
    best_epochs = []
    for meta_test_batch in tqdm(meta_task_data_loader.tasks(), total=len(meta_task_data_loader)):
        for task in tqdm(meta_test_batch):
            # pdb.set_trace() # look at NVIDIA memory usage here
            results, best_epoch = _meta_test_on_task(maml_model, task, meta_test_epochs, loss_func, metric_func, dataset_type, args, save_dir, logger)
            best_epochs.append(best_epoch)
            test_task_results.append(results)
            torch.cuda.empty_cache()

    return test_task_results, best_epochs