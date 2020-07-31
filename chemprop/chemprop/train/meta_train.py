import logging
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.data.utils import DataLoader
from tqdm import tqdm, trange
import numpy as np

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset, TaskDataLoader
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR
import wandb
from memory_profiler import profile
import pdb

def predict_on_batch_and_return_loss(learner, batch, task_idx, loss_func):
    mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()
    batch_targets = torch.Tensor([tb[task_idx] for tb in target_batch])
    preds = learner(mol_batch, features_batch)

    if preds.size()[-1] == 1:
        preds = torch.squeeze(preds, dim = -1)
    # Move tensors to correct device and calculate average loss
    batch_targets = batch_targets.to(preds.device)
    loss = loss_func(preds, batch_targets)
    loss = loss.mean()
    return loss

def get_task_idx(task):
    return np.argmax(task.get_task_mask())

def fast_adapt(learner, task, task_idx, loss_func, num_inner_steps):
    """
    Takes care of logic for fast adapting the learner
    """
    # pdb.set_trace()
    # Fast adapt learner over batches of this task
    learner.train()

    # Calling iter effectively shuffles the data so that we get a new ordering of batches, effectively a sample of two new batches
    train_data_loader_iterator = iter(task.train_data_loader)
    for step in range(num_inner_steps):
        batch = next(train_data_loader_iterator)

        adaptation_loss = predict_on_batch_and_return_loss(learner, batch, task_idx, loss_func)
        learner.adapt(adaptation_loss)
        # wandb.log({'{}_adaptation_loss'.format(task.assay_name): adaptation_loss})

def calculate_meta_loss(learner, task, task_idx, loss_func):
    # After inner adaptation steps, calculate evaluation loss and return 

    # Calling iter effectively allows us to fetch a random sample from the val data loader since the val data loader always has shuffle=True
    batch = next(iter(task.val_data_loader))
    eval_loss = predict_on_batch_and_return_loss(learner, batch, task_idx.item(), loss_func)
    
    return eval_loss 

# @profile
def meta_train(maml_model,
          meta_task_data_loader: DataLoader,
          epoch: int,
          loss_func: Callable,
          loss_queue,
          meta_optimizer: Optimizer,
          args: TrainArgs,
          logger: logging.Logger = None) -> int:
    """
    Trains a model for an epoch on TASKS.
    We define an epoch as a loop over all batches of tasks.
    This means we cycle through all batches of tasks, and on each task, we take a certain number of fixed inner gradient steps.

    The number of inner gradient steps represents the number of batches per task the model fast adapts on.

    After each batch of inner task fast adaptations, an outer loop update will be performed.

    The epoch concludes once all task batches have been cycled through.

    :param maml_model: l2L maml Model.
    :param meta_task_data_loader: A DataLoader over tasks.
    :param epoch: The current epoch -- used to skip batches of data for each task we've already seen while preserving the iterator interface
    :param loss_func: Loss function.
    :param meta_optimizer: An Optimizer.
    :param args: Arguments.
    :param logger: A logger for printing intermediate results.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print
    maml_model.train()
    # meta_train_error refers to the fast adaptation error on each task
    # meta_val_error refers to the error on the evaluation sets from each task
    running_loss = 0.0
    num_meta_iters = 0
    for meta_train_batch in tqdm(meta_task_data_loader.tasks(), total = len(meta_task_data_loader)):
        
        task_evaluation_loss = 0.0
        
        for task in tqdm(meta_train_batch):
            learner = maml_model.clone()
            curr_task_target_idx = get_task_idx(task)
            
            fast_adapt(learner, task, curr_task_target_idx, loss_func, args.num_inner_gradient_steps)
            eval_loss = calculate_meta_loss(learner, task, curr_task_target_idx, loss_func)
            task_evaluation_loss += eval_loss

        # Average the task evaluation loss over the size of the batch
        task_evaluation_loss = task_evaluation_loss / len(meta_train_batch)
        running_loss += task_evaluation_loss.item() # add average batch loss to running loss
        num_meta_iters += 1
        # Now that we are done with meta batch of tasks, perform meta update.
        # Zero out the meta opt gradient for new meta batch
        meta_optimizer.zero_grad()
        task_evaluation_loss.backward()
        # gradient clip
        torch.nn.utils.clip_grad_norm_(maml_model.parameters(), 1., norm_type=2)
        # take optimizer step
        meta_optimizer.step()

        # Compute stats and log to wandb 
        with torch.no_grad():
            avg_meta_loss = task_evaluation_loss.item()
        
        if len(loss_queue) >= args.loss_queue_window:
            loss_queue.popleft()
        loss_queue.append(avg_meta_loss)
        curr_loss_average = np.mean(list(loss_queue))
        pnorm = compute_pnorm(maml_model)
        gnorm = compute_gnorm(maml_model)
        debug(f'Meta loss on this task batch = {avg_meta_loss:.4e}, Meta loss averaged over last {args.loss_queue_window} steps = {curr_loss_average:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}')

        wandb.log({'meta_train_batch_loss': avg_meta_loss, 'meta_loss_{}_window_avg'.format(args.loss_queue_window): curr_loss_average, 'PNorm': pnorm, 'GNorm': gnorm})
    average_loss_across_meta_batches = running_loss/num_meta_iters
    return average_loss_across_meta_batches