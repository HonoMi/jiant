import os
import argparse
import logging

import jiant.utils.python.io as py_io
import jiant.scripts.download_data.dl_datasets.hf_datasets_tasks as hf_datasets_tasks_download
# import jiant.scripts.download_data.dl_datasets.xtreme as xtreme_download  # XXX: 20201204 変なエラーを吐くので，実験期間中は一旦，排除する．
xtreme_download = None
import jiant.scripts.download_data.dl_datasets.files_tasks as files_tasks_download
from jiant.tasks.constants import (
    GLUE_TASKS,
    SUPERGLUE_TASKS,
    XTREME_TASKS,
    BENCHMARKS,
)
from jiant.scripts.download_data.constants import (
    SQUAD_TASKS,
    DIRECT_DOWNLOAD_TASKS,
    OTHER_HF_DATASETS_TASKS,
)
from jiant.scripts.download_data.cross_validation import parse_cv_task_name

# DIRECT_DOWNLOAD_TASKS need to be directly downloaded because the HF Datasets
# implementation differs from the original dataset format
HF_DATASETS_TASKS = (GLUE_TASKS | SUPERGLUE_TASKS | OTHER_HF_DATASETS_TASKS) - DIRECT_DOWNLOAD_TASKS
SUPPORTED_TASKS = HF_DATASETS_TASKS | XTREME_TASKS | SQUAD_TASKS | DIRECT_DOWNLOAD_TASKS

logger = logging.getLogger(__name__)


# noinspection PyUnusedLocal
def list_supported_tasks_cli(args):
    print("Supported tasks:")
    for task in sorted(list(SUPPORTED_TASKS)):
        print(task)


def download_data_cli(args):
    output_base_path = args.output_path
    if args.tasks:
        task_names = args.tasks
    elif args.benchmark:
        task_names = BENCHMARKS[args.benchmark]
    else:
        raise RuntimeError()
    download_data(
        task_names=task_names, output_base_path=output_base_path,
    )


def download_data(task_names, output_base_path):
    task_data_base_path = py_io.create_dir(output_base_path, "data")
    task_config_base_path = py_io.create_dir(output_base_path, "configs")

    # assert set(task_names).issubset(SUPPORTED_TASKS)
    task_names_with_cv = task_names

    # Download specified tasks and generate configs for specified tasks
    for i, cv_task_name in enumerate(task_names_with_cv):
        task_name, n_fold, fold = parse_cv_task_name(cv_task_name)

        assert(task_name in SUPPORTED_TASKS)

        task_data_path = os.path.join(task_data_base_path, cv_task_name)

        if task_name in HF_DATASETS_TASKS:
            hf_datasets_tasks_download.download_data_and_write_config(
                task_name = task_name,
                task_data_path = task_data_path,
                task_config_path = os.path.join(task_config_base_path, f"{cv_task_name}_config.json"),
                n_fold=n_fold,
                fold=fold,
            )
        elif task_name in XTREME_TASKS:
            if n_fold is not None:
                raise NotImplementedError()
            xtreme_download.download_xtreme_data_and_write_config(
                task_name=task_name,
                task_data_base_path=task_data_base_path,
                task_config_base_path=task_config_base_path,
            )
        elif task_name in DIRECT_DOWNLOAD_TASKS:
            if n_fold is not None:
                raise NotImplementedError()
            files_tasks_download.download_task_data_and_write_config(
                task_name=task_name,
                task_data_path=task_data_path,
                task_config_path=os.path.join(task_config_base_path, f"{cv_task_name}_config.json"),
            )
        else:
            raise KeyError()
        logger.info(f"Downloaded and generated configs for '{cv_task_name}' ({i+1}/{len(task_names)})")


def main():
    parser = argparse.ArgumentParser(description="Download datasets and generate task configs")
    subparsers = parser.add_subparsers()
    sp_list = subparsers.add_parser("list", help="list supported tasks in downloader")
    sp_download = subparsers.add_parser("download", help="download data command")
    sp_download.add_argument(
        "--output_path", required=True, help="base output path for downloaded data and task configs"
    )
    sp_download_group = sp_download.add_mutually_exclusive_group(required=True)
    sp_download_group.add_argument("--tasks", nargs="+", help="list of tasks to download")
    sp_download_group.add_argument("--benchmark", choices=BENCHMARKS)

    # Hook subparsers up to functions
    sp_list.set_defaults(func=list_supported_tasks_cli)
    sp_download.set_defaults(func=download_data_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
