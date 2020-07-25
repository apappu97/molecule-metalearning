import os
import logging
from typing import Callable, List
import torch
import torch.nn as nn
from tqdm import tqdm, trange

from .predict import predict
from .evaluate import evaluate_predictions
from chemprop.data import MoleculeDataLoader, StandardScaler
from chemprop.data import MetaTaskDataLoader, TaskDataLoader
from .meta_train import fast_adapt, process_batch_and_predict
from chemprop.utils import save_checkpoint, load_checkpoint
import wandb
from memory_profiler import profile
import pdb

@profile
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
            learner = maml_model.clone()
            learner.train()
            curr_task_mask = torch.Tensor(task.get_task_mask()) # Bit vector for current task
            curr_task_target_idx = torch.argmax(curr_task_mask)
            fast_adapt(learner, task, curr_task_target_idx, loss_func, num_inner_gradient_steps)

            # Evaluate predictions now 
            learner.eval()
            preds = predict(
                model=learner,
                data_loader=task.val_data_loader,
                scaler=None
            )

            targets = task.get_targets('val')

            results = evaluate_predictions(
                preds=preds,
                targets=targets,
                num_tasks=1,
                metric_func=metric_func,
                dataset_type=dataset_type,
                logger=logger
            )

            val_task_results.append(results)

    return val_task_results

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
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print
    test_task_results = []
    
    best_epochs = []
    for meta_test_batch in tqdm(meta_task_data_loader.tasks(), total=len(meta_task_data_loader)):
        for task in tqdm(meta_test_batch):
            info("Meta testing on task: {}".format(task.assay_name))
            learner = maml_model.clone()
            curr_task_mask = torch.Tensor(task.get_task_mask()) # Bit vector for current task
            curr_task_target_idx = torch.argmax(curr_task_mask)
            
            best_val_loss = float('inf')
            best_epoch = 0
            # save the best performing model in case it occurred at 0 epochs 
            save_checkpoint(os.path.join(save_dir, 'meta_test_{}_model.pt'), learner, scaler=None, features_scaler=None, args=args)

            for epoch in trange(meta_test_epochs):
                task_adaptation_loss = 0.0
                learner.train()
                for batch in task.train_data_loader:
                    preds, batch_targets = process_batch_and_predict(learner, batch, curr_task_target_idx.item())
                    adaptation_loss = loss_func(preds, batch_targets)
                    adaptation_loss = adaptation_loss.mean()
                    learner.adapt(adaptation_loss)
                    with torch.no_grad():
                        batch_avg_loss = adaptation_loss.item()
                    wandb.log({'meta_test_{}_adaptation_loss'.format(task.assay_name): batch_avg_loss})
                    task_adaptation_loss += batch_avg_loss
                    del preds
                    del batch_targets
                    del adaptation_loss
                
                task_adaptation_loss /= len(task.train_data_loader) # normalize by number of batches to get avg batch mean loss 
                wandb.log({'meta_test_{}_epoch_adaptation_loss'.format(task.assay_name): task_adaptation_loss})
                
                # validation 
                task_val_loss = 0.0
                learner.eval()
                with torch.no_grad():
                    for batch in task.val_data_loader:
                        preds, batch_targets = process_batch_and_predict(learner, batch, curr_task_target_idx.item())
                        adaptation_loss = loss_func(preds, batch_targets)
                        batch_avg_loss = adaptation_loss.mean().item()
                        task_val_loss += batch_avg_loss
                
                task_val_loss /= len(task.val_data_loader) # normalize by number of batches to get avg batch loss
                wandb.log({'meta_test_{}_epoch_val_loss'.format(task.assay_name): task_val_loss})
                
                if task_val_loss < best_val_loss:
                    info('New best model for test task {} at epoch {} with loss {}'.format(task.assay_name, epoch + 1, task_val_loss))
                    best_val_loss = task_val_loss
                    best_epoch = epoch
                    # save best checkpoint at meta test time 
                    save_checkpoint(os.path.join(save_dir, 'meta_test_{}_model.pt'.format(task.assay_name)), learner, args=args)        

            best_epochs.append(best_epoch)
            info('Finished early stopping for task {}, beginning testing'.format(task.assay_name))
            # Now that early stopping has identified the best model, calculate test loss
            pdb.set_trace()
            model = load_checkpoint(os.path.join(save_dir, 'meta_test_{}_model.pt'.format(task.assay_name)))
            model.eval()
            preds = predict(
                model=model,
                data_loader=task.test_data_loader,
                scaler=None
            )

            targets = task.get_targets('test')

            results = evaluate_predictions(
                preds=preds,
                targets=targets,
                num_tasks=1,
                metric_func=metric_func,
                dataset_type=dataset_type,
                logger=logger
            )
            test_task_results.append(results)

            del model # no longer need the model for this test task

    return test_task_results, best_epochs
