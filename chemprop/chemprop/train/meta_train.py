import logging
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset, MetaTaskDataLoader, TaskDataLoader
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def meta_train(maml_model,
          meta_task_data_loader: MetaTaskDataLoader,
          loss_func: Callable,
          meta_optimizer: Optimizer,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch on TASKS.
    We define an epoch as a loop over all batches of tasks.
    This means we cycle through all batches of tasks, and on each task, we take a certain number of fixed inner gradient steps.

    The number of inner gradient steps represents the number of batches per task the model fast adapts on.

    After each batch of inner task fast adaptations, an outer loop update will be performed.

    The epoch concludes once all task batches have been cycled through.

    :param maml_model: l2L maml Model.
    :param data_loader: A MoleculeDataLoader.
    :param loss_func: Loss function.
    :param meta_optimizer: An Optimizer.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print
    maml_model.train()
    # meta_train_erorr refers to the fast adaptation error on each task
    # meta_val_error refers to the error on the evaluation sets from each task
    total_meta_train_error, total_meta_val_error = 0.0, 0.0

    import pdb; pdb.set_trace()
    for meta_train_batch in tqdm(train_meta_task_data_loader.tasks(), total = len(train_meta_task_data_loader)):
        # Zero out the meta opt gradient for new meta batch
        meta_optimizer.zero_grad()
        evaluation_losses = []
        for task in tqdm(meta_train_batch):
            learner = maml_model.clone()
            curr_task_mask = torch.Tensor(task.get_task_mask()) # Bit vector for current task
            curr_task_target_idx = torch.argmax(curr_task_mask)
            curr_inner_gradient_step = 0 
            # Fast adapt learner over batches of this task
            for batch in tqdm(task.train_data_loader, total=args.num_inner_gradient_steps):
                mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()
                batch_targets = torch.Tensor(target_batch[:, curr_task_target_idx])
                preds = learner(mol_batch, features_batch)
                rel_preds = preds[:, curr_task_target_idx]
                assert rel_preds.size() == batch_targets.size()

                # Move tensors to correct device
                batch_targets = batch_targets.to(preds.device)
                adaptation_loss = loss_func(preds, batch_targets)
                learner.adapt(adaptation_loss)

                curr_inner_gradient_step += 1
                if curr_inner_gradient_step>= args.num_inner_gradient_steps: break

           # After inner adaptation steps, calculate evaluation loss and store for later meta update (after batch of tasks)
            for batch in tqdm(task.val_data_loader):
                mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()
                batch_targets = torch.Tensor(target_batch[:, curr_task_target_idx])
                preds = learner(mol_batch, features_batch)

                # Move tensors to correct device
                batch_targets = batch_targets.to(preds.device)
                eval_loss = loss_func(preds, batch_targets)
                # Call backward here according to l2l documentation
                eval_loss.backward()
                evaluation_losses.append(eval_loss.item())
                break

        # Now done with meta batch of tasks, perform meta update.
        evaluation_loss = sum(evaluation_losses)
        meta_optimizer.step()
        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)

    return n_iter
