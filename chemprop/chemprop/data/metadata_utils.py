from .data import MoleculeDatapoint, MoleculeDataset, MoleculeDataLoader
from .utils import split_data
from .scaffold import scaffold_split
from typing import Callable, Dict, Iterator, List, Union, Tuple
import torch #type: ignore
from logging import Logger
from chemprop.args import TrainArgs
import numpy as np
import math

class TaskDataLoader:
    """ A TaskDataLoader is a wrapper for a single task, and handles generating
    the train, validation (optional) and test sets for this task """

    def __init__(self,
            dataset: MoleculeDataset,
            task_mask: List[int],
            split_type: str = 'random',
            sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            num_workers: int = 8,
            cache: bool = False,
            args: TrainArgs = None,
            logger: Logger = None):
        """
        Initializes a TaskDataLoader
        An important note on batch size: The batch size represents the k
        instances to be sampled from each task in the fast_adapt phase of
        learning.

        :param dataset: A MoleculeDataset
        :param split_type: type of split
        :param sizes: tuple for size of splits
        :param task_mask: A one-hot vector of length num_tasks indicating the target
        :param num_works: Number of workers used to build batches
        :param cache: Whether to cache the graph featurizations of molecules
        for faster processing
        :param args: TrainArgs
        :param logger: logger for logging
        """
        if not np.isclose(sum(sizes), 1):
            raise ValueError('Size of splits must sum to 1!')
        if len(task_mask) != dataset.num_tasks():
            raise ValueError('Task mask is a one-hot vector, so should be equal to num_tasks() on MoleculeDataset object')

        self.task_mask = task_mask
        task_idx = np.argmax(task_mask)
        self.task_idx = task_idx
        # Retrieve molecules that have an entry for this task, as there may be
        # missing data
        targets = dataset.targets()
        task_dataset = []
        # Verify that the dataset[i].mol.targets() has the same output as
        # targets[i] for ordering purposes
        for i in range(len(dataset)):
            if targets[i][task_idx] is not None:
                task_dataset.append(dataset[i])
	
        self.data = MoleculeDataset(task_dataset)
        # Now that we have relevant items, split accordingly
        # import pdb; pdb.set_trace()
        if split_type == 'random':
            self.data.shuffle(seed=args.seed)
            train_size = int(sizes[0] * len(self.data))
            if sizes[1] != 0:
                train_val_size = int((sizes[0] + sizes[1]) * len(self.data))
                train = self.data[:train_size]
                val = self.data[train_size:train_val_size]
                test = self.data[train_val_size:]

                train_data = MoleculeDataset(train)
                val_data = MoleculeDataset(val)
                test_data = MoleculeDataset(test)

            else:
                # no holdout set
                train = self.data[:train_size]
                val = None
                test = self.data[train_size:]

                train_data = MoleculeDataset(train)
                test_data = MoleculeDataset(test)
        elif split_type == 'scaffold_balanced':
            if sizes[1] == 0:
                val = None
            else:
                # This is hacky
                val = 'Not None'
            train_data, val_data, test_data = scaffold_split(self.data, sizes=sizes, balanced=True, seed=args.seed, logger=logger)

        if args.features_scaling:
            features_scaler = train_data.normalize_features(replace_nan_token=0)
            if val is not None:
                val_data.normalize_features(features_scaler)
            test_data.normalize_features(features_scaler)
        else:
            features_scaler = None

        self.features_scaler = features_scaler

        self._train_data_loader = MoleculeDataLoader(
                dataset=train_data,
                batch_size=args.batch_size,
                num_workers=num_workers,
                cache=cache,
                class_balance=args.class_balance,
                shuffle=True,
                seed=args.seed)

        if val is not None:
            self._val_data_loader = MoleculeDataLoader(
                    dataset=val_data,
                    batch_size=args.batch_size,
                    num_workers=num_workers,
                    cache=cache)
        else:
            self._val_data_loader = None

        self._test_data_loader = MoleculeDataLoader(
                dataset=test_data,
                batch_size=args.batch_size,
                num_workers=num_workers,
                cache=cache)

    @property
    def train_data_loader(self) -> MoleculeDataLoader:
        return self._train_data_loader

    @property
    def val_data_loader(self) -> MoleculeDataLoader:
        # Note, this may be none
        return self._val_data_loader

    @property
    def test_data_loader(self) -> MoleculeDataLoader:
        return self._test_data_loader

    def get_task_mask(self) -> torch.Tensor:
        return self.task_mask

class MetaTaskDataLoader:
    """
    A wrapper for a set of tasks, possibly representing a meta train task
    split, val task split, etc.

    """
    def __init__(self,
            dataset: MoleculeDataset,
            tasks: List[int],
            sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            num_workers: int = 8,
            cache: bool = False,
            args: TrainArgs = None,
            logger: Logger = None):
        """
        Initializes a MetaTaskDataLoader

        :param dataset: A MoleculeDataset
        :param split_type: type of split
        :param sizes: tuple for size of splits
        :param tasks: Bit vector with 1s for tasks
        :param num_works: Number of workers used to build batches
        :param cache: Whether to cache the graph featurizations of molecules
        for faster processing
        :param args: TrainArgs
        :param logger: logger for logging
        """
        if len(tasks) != dataset.num_tasks():
            raise ValueError("length of tasks must equal dataset.num_tasks()")
        self.meta_batch_size = args.meta_batch_size
        task_data_loaders = []
        for idx in np.nonzero(tasks)[0]:
            task_mask = [0] * len(tasks)
            task_mask[idx] = 1
            task_data_loader = TaskDataLoader(
                    dataset=dataset,
                    task_mask=tasks,
                    split_type=args.split_type,
                    sizes=sizes,
                    num_workers=num_workers,
                    cache=cache,
                    args=args,
                    logger=logger)
            task_data_loaders.append(task_data_loader)
        self.task_data_loaders = task_data_loaders

#    def __iter__(self) -> Iterator[TaskDataLoader]:
#        """
#        Returns an iterator over the task data loaders
#        """
#        return iter(self.task_data_loaders)


    def tasks(self) -> Iterator[TaskDataLoader]:
        """
        Generator for iterating through tasks
        """
        for i in range(0, len(self.task_data_loaders), self.meta_batch_size):
            yield self.task_data_loaders[i:i+self.meta_batch_size]

    def __len__(self) -> int:
        """ 
        Returns number of task batches
        """
        return math.ceil(len(self.task_data_loaders) * 1.0 / self.meta_batch_size)
