import os
import logging
from typing import List

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.torch_utils as torch_utils

from torchsampler import WeightedDatasetSampler

logger = logging.getLogger(__name__)


class _ListDataset(Dataset):
    def __init__(self, elems: List):
        self._elems = elems

    def __getitem__(self, index):
        return self._elems[index]
        raise NotImplementedError

    def __len__(self):
        return len(self._elems)


def complex_backpropagate(
    loss, optimizer, model, fp16, n_gpu, gradient_accumulation_steps, max_grad_norm
):
    if n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps
    if fp16:
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from apex import amp

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    return loss


def get_train_dataloader_from_cache(
    train_cache: caching.ChunkedFilesDataCache, task, train_batch_size: int,
    sample_weights_path=None,
    fix_seed_for_weighted_sampler=False,
):
    # TODO: Expose buffer_size parameter  (issue #1183)

    if sample_weights_path is not None:
        dataset = train_cache.get_iterable_dataset(buffer_size=10000, shuffle=False)
        dataset = _ListDataset([elem for elem in dataset])
        _sample_weights = pd.read_csv(sample_weights_path, sep='\t', header=None)[0]
        sampler = WeightedDatasetSampler(dataset, _sample_weights, fix_seed=fix_seed_for_weighted_sampler)
    else:
        dataset = train_cache.get_iterable_dataset(buffer_size=10000, shuffle=True)
        sampler = None

    train_dataloader = torch_utils.DataLoaderWithLength(
        dataset=dataset, batch_size=train_batch_size, collate_fn=task.collate_fn,
        sampler=sampler
    )
    return train_dataloader


def get_eval_dataloader_from_cache(
    eval_cache: caching.ChunkedFilesDataCache,
    task,
    eval_batch_size: int,
    subset_num=None,
    explicit_subset=None,
):
    dataset = eval_cache.get_iterable_dataset(
        buffer_size=10000, shuffle=False, subset_num=subset_num, explicit_subset=explicit_subset,
    )
    eval_dataloader = torch_utils.DataLoaderWithLength(
        dataset=dataset, batch_size=eval_batch_size, collate_fn=task.collate_fn,
    )
    return eval_dataloader


def save_model_with_metadata(model: nn.Module, metadata: dict, output_dir: str, file_name="model"):
    torch.save(
        torch_utils.get_model_for_saving(model).state_dict(),
        os.path.join(output_dir, f"{file_name}.p"),
    )
    py_io.write_json(metadata, os.path.join(output_dir, f"{file_name}.metadata.json"))


def compare_steps_max_steps(step, max_steps):
    return max_steps is not None and max_steps != -1 and step >= max_steps
