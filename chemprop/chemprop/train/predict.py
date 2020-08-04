from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler


def predict(model: nn.Module,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None,
            return_targets: bool = False,
            task_idx: int = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data_loader: A MoleculeDataLoader.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    if return_targets and not task_idx:
        raise ValueError("If targets are desired must pass in task index")
    model.eval()

    preds = []
    if return_targets:
        targets = []
    for batch in tqdm(data_loader, disable=disable_progress_bar):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch = batch.batch_graph(), batch.features()
        if return_targets:
            targets_batch = batch.targets()
            for t in targets_batch:
                targets.append([t[task_idx]])

        # Make predictions
        with torch.no_grad():
            batch_preds = model(mol_batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    if return_targets:
        return preds, targets 
    else:
        return preds
    