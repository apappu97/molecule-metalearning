from .data import MoleculeDatapoint, MoleculeDataset, MoleculeDataLoader
from .utils import split_data
from .scaffold import scaffold_split
from typing import Callable, Dict, Iterator, List, Union, Tuple
import torch #type: ignore
from logging import Logger
from chemprop.args import TrainArgs
import numpy as np
import math
from tqdm import tqdm
from memory_profiler import profile
import pdb

class TaskDataLoader:
    """ 
    A TaskDataLoader is a wrapper for a single task, and handles generating
    the train, validation and test (optional) datasets for this task.

    At meta train time, we only need a train (fast adapt) and validation set (meta update on validation set)
    At meta test time, we need a train (fast adapt), validation (early stopping), and test set (for testing)
    """

    def __init__(self,
            dataset: MoleculeDataset,
            assay_name: str,
            task_mask: List[int],
            num_workers: int,
            split_type: str = 'random',
            sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
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

        self._assay_name = assay_name
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
            if sizes[2] != 0:
                # At meta test time, so need a test set
                train_val_size = int((sizes[0] + sizes[1]) * len(self.data))
                train = self.data[:train_size]
                val = self.data[train_size:train_val_size]
                test = self.data[train_val_size:]

                train_data = MoleculeDataset(train)
                val_data = MoleculeDataset(val)
                test_data = MoleculeDataset(test)

            else:
                # At meta train time, no test set needed
                train = self.data[:train_size]
                val = self.data[train_size:]
                test = None 

                train_data = MoleculeDataset(train)
                val_data = MoleculeDataset(test)
        elif split_type == 'scaffold_balanced':
            # This is hacky
            if sizes[2] == 0:
                test = None
            else:
                test = 'Not None'
            # if assay_name == "CHEMBL1243965":
            #     pdb.set_trace()
            train_data, val_data, test_data = scaffold_split(self.data, sizes=sizes, balanced=True, seed=args.seed, logger=logger)

        if args.features_scaling:
            features_scaler = train_data.normalize_features(replace_nan_token=0)
            val_data.normalize_features(features_scaler)

            if test is not None:
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
        self._train_data_loader_iterator = iter(self._train_data_loader)

        self._val_data_loader = MoleculeDataLoader(
                dataset=val_data,
                batch_size=args.batch_size,
                num_workers=num_workers,
                cache=cache)
        self._val_data_loader_iterator = iter(self._val_data_loader)

        if test is not None:
            self._test_data_loader = MoleculeDataLoader(
                dataset=test_data,
                batch_size=args.batch_size,
                num_workers=num_workers,
                cache=cache)
            self._test_data_iterator = iter(self._test_data_loader)
        else:
            self._test_data_loader = None 
            self._test_data_iterator = None 

    @property 
    def train_data_loader(self):
        return self._train_data_loader

    @property 
    def val_data_loader(self):
        return self._val_data_loader

    @property
    def test_data_loader(self):
        return self._test_data_loader

    @property
    def train_data_loader_iter(self):
        return self._train_data_loader_iterator

    @property
    def val_data_loader_iter(self):
        # Note, this may be none
        return self._val_data_loader_iterator

    @property
    def test_data_loader_iter(self):
        return self._test_data_loader_iterator
    
    def get_targets(self, split):
        """
        Return list of targets for THIS task for relevant split
        """
        assert split in ['train', 'val', 'test']
        if split == 'train':
            data_loader = self._train_data_loader
        elif split == 'val':
            data_loader = self._val_data_loader
        else:
            data_loader = self._test_data_loader
        all_targets = data_loader.targets()
        targets = []
        for t in all_targets:
            targets.append([t[self.task_idx]])
        return targets
    
    def re_initialize_iterator(self, split):
        """
        A bit hacky, re initializes the iterator when it's been exhausted
        """
        assert split in ['train', 'val', 'test']
        if split == 'train':
            self._train_data_loader_iterator = iter(self._train_data_loader)
        elif split == 'val':
            self._val_data_loader_iterator = iter(self._val_data_loader)
        else:
            self._test_data_loader_iterator = iter(self._test_data_loader)

    @property
    def assay_name(self) -> str:
        return self._assay_name

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
            task_names: List[str],
            meta_batch_size: int,
            num_workers: int,
            sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            cache: bool = False,
            args: TrainArgs = None,
            logger: Logger = None):
        """
        Initializes a MetaTaskDataLoader

        :param dataset: A MoleculeDataset
        :param split_type: type of split
        :param sizes: tuple for size of splits
        :param tasks: Bit vector with 1s for all tasks in this split
        :param num_works: Number of workers used to build batches
        :param cache: Whether to cache the graph featurizations of molecules
        for faster processing
        :param args: TrainArgs
        :param logger: logger for logging
        """
        if len(tasks) != dataset.num_tasks():
            raise ValueError("Length of tasks must equal dataset.num_tasks()")
        self.meta_batch_size = meta_batch_size

        task_data_loaders = []
        meta_task_names = []
        for idx in np.nonzero(tasks)[0]:
            task_mask = [0] * len(tasks)
            task_mask[idx] = 1
            task_name = task_names[idx]
            meta_task_names.append(task_name)
            task_data_loader = TaskDataLoader(
                    dataset=dataset,
                    assay_name=task_name,
                    task_mask=task_mask,
                    num_workers=num_workers,
                    split_type=args.split_type,
                    sizes=sizes,
                    cache=cache,
                    args=args,
                    logger=logger)
            task_data_loaders.append(task_data_loader)
        self.task_data_loaders = task_data_loaders
        self.meta_task_names = meta_task_names

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
