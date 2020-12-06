import json
import os
import numpy as np
import logging

import torch
import pandas as pd

import jiant.utils.python.io as py_io
import jiant.proj.main.components.task_sampler as jiant_task_sampler

from machine_learning.dataset.preprocessing.pandas_dataframe import jsonify as jsonify_df

logger = logging.getLogger(__name__)


def write_train_eval_results(results_dict, metrics_aggregator, output_dir, verbose=True):
    _write_results("train", results_dict, metrics_aggregator, output_dir, verbose=verbose)


def write_val_results(results_dict, metrics_aggregator, output_dir, verbose=True):
    _write_results("val", results_dict, metrics_aggregator, output_dir, verbose=verbose)


def write_test_results(results_dict, metrics_aggregator, output_dir, verbose=True):
    _write_results("test", results_dict, metrics_aggregator, output_dir, verbose=verbose)
    

def _write_results(split: str, results_dict, metrics_aggregator, output_dir, verbose=True):
    full_results_to_write = {
        "aggregated": jiant_task_sampler.compute_aggregate_major_metrics_from_results_dict(
            metrics_aggregator=metrics_aggregator, results_dict=results_dict,
        ),
    }
    for task_name, task_results in results_dict.items():
        task_results_to_write = {}
        if "loss" in task_results:
            task_results_to_write["loss"] = task_results["loss"]
        if "metrics" in task_results:
            task_results_to_write["metrics"] = task_results["metrics"].to_dict()
        full_results_to_write[task_name] = task_results_to_write

    metrics_str = json.dumps(full_results_to_write, indent=2)
    if verbose:
        logger.info(metrics_str)

    py_io.write_json(data=full_results_to_write, path=os.path.join(output_dir, f"{split}_metrics.json"))


def write_preds(eval_results_dict, path):
    preds_dict = {}
    df_dict = {
        "preds": [],
        "guids": [],
        "task": [],
        "logits": [],
    }

    for task_name, task_results_dict in eval_results_dict.items():
        preds = task_results_dict["preds"]
        guids = task_results_dict["accumulator"].get_guids()
        logits = task_results_dict.get("logits", [np.nan] * len(preds))
        tasks = [task_name] * len(preds)

        preds_dict[task_name] = {
            "preds": preds,
            "guids": guids,
            "logits": logits,
        }

        df_dict['preds'].extend(preds)
        df_dict['guids'].extend(guids)
        df_dict['logits'].extend(logits)
        df_dict['task'].extend(tasks)

    torch.save(preds_dict, path)

    df = pd.DataFrame(df_dict)
    jsonify_df(df).to_csv(path.rstrip('.p') + '.tsv', sep='\t', index=None)
