import os
import tarfile
import urllib
import zipfile

from datasets import Dataset, DatasetDict, Value, Metric
import jiant.utils.python.io as py_io
from jiant.utils.python.datastructures import replace_key
from datasets_extra.processing import build_cv_splits
from datasets_extra import load_metric, load_dataset
# from datasets_extra import load_dataset as load_hf_dataset, load_metric as load_hf_metric


def load_hf_dataset(path,
                    name=None,
                    version=None,
                    phase_map=None,
                    n_fold: int = None,
                    fold: int = None,
                    split_type: str = None):
    phase_map = phase_map or {}
    split_type = split_type or 'local_evaluation'

    dataset_dict = load_dataset(path=path, name=name, version=version)

    for old_phase_name, new_phase_name in phase_map.items():
        replace_key(dataset_dict, old_key=old_phase_name, new_key=new_phase_name)

    if n_fold is not None:
        jiant2hf_phase_map = {val: key for key, val in phase_map.items()}
        hf_train_phase = jiant2hf_phase_map.get('train', 'train')
        hf_val_phase = jiant2hf_phase_map.get('val', 'val')

        if split_type == 'local_evaluation':
            hf_local_phase = '+'.join([hf_train_phase])
            dataset_dict['test'] = dataset_dict['val']
        elif split_type == 'submission':
            hf_local_phase = '+'.join([hf_train_phase, hf_val_phase])
        else:
            raise ValueError(f'Unknown split type "{split_type}"')

        hf_local_dataset = load_dataset(path=path,
                                        name=name,
                                        version=version,
                                        split=hf_local_phase)
        cv_dataset = build_cv_splits(
            hf_local_dataset,
            n_fold,
            stratify=False,
        )
        dataset_dict['train'] = cv_dataset[f'fold-{fold}.train']
        dataset_dict['val'] = cv_dataset[f'fold-{fold}.val']

    return dataset_dict


def convert_hf_dataset_to_examples(
    path, name=None, version=None, field_map=None, label_map=None, phase_map=None, phase_list=None,
    n_fold: int = None, fold: int = None,
    return_all: bool = False,
    experiment_id_for_metric=None,
    cache_dir_for_metric=None,
):
    """Helper function for reading from datasets.load_dataset and converting to examples

    Args:
        path: path argument (from datasets.load_dataset)
        name: name argument (from datasets.load_dataset)
        version: version argument (from datasets.load_dataset)
        field_map: dictionary for renaming fields, non-exhaustive
        label_map: dictionary for replacing labels, non-exhaustive
        phase_map: dictionary for replacing phase names, non-exhaustive
        phase_list: phases to keep (after phase_map)

    Returns:
        Dict[phase] -> list[examples]
    """
    # "mrpc.cv-5-0"
    fold_dataset = load_hf_dataset(path=path,
                                   name=name,
                                   version=version,
                                   phase_map=phase_map,
                                   n_fold=n_fold,
                                   fold=fold)

    if phase_list is None:
        phase_list = fold_dataset.keys()
    examples_dict = {}
    for phase in phase_list:
        phase_examples = []
        for raw_example in fold_dataset[phase]:
            if field_map:
                for old_field_name, new_field_name in field_map.items():
                    replace_key(raw_example, old_key=old_field_name, new_key=new_field_name)
            if label_map and "label" in raw_example:
                # Optionally use an dict or function to map labels
                label = raw_example["label"]

                if isinstance(label_map, dict):
                    if raw_example["label"] in label_map:
                        label = label_map[raw_example["label"]]
                elif callable(label_map):
                    label = label_map(raw_example["label"])
                else:
                    raise TypeError(label_map)

                raw_example["label"] = label
            phase_examples.append(raw_example)
        examples_dict[phase] = phase_examples

    metric = load_metric(path,
                         config_name=name,
                         experiment_id=experiment_id_for_metric,
                         cache_dir=cache_dir_for_metric)

    # label to integer
    # huggingface-datasetsはたまに，"neutral"などのラベルを持ったデータセットを返す．
    # これを整数にマップする．
    inverse_label_map = {}
    if label_map is None:
        pass
    elif isinstance(label_map, dict):
        inverse_label_map = {val: key for key, val in label_map.items()}
    else:
        raise NotImplementedError()

    fold_dataset_with_integer_labels = fold_dataset.copy()
    for name, dataset in fold_dataset.items():
        samples = dataset[:].copy()
        features = dataset.features.copy()

        if 'label' not in samples:
            continue

        for i in range(0, len(samples['label'])):
            label = samples['label'][i]
            samples['label'][i] = inverse_label_map.get(label, label)
        features['label'] = Value('int64')
        _dataset = Dataset.from_dict(samples, features=features)
        fold_dataset_with_integer_labels[name] = _dataset

    if return_all:
        return examples_dict, {'hf_fold_dataset': fold_dataset_with_integer_labels,
                               # 'hf_full_dataset': full_dataset,
                               'hf_metric': metric}
    else:
        return examples_dict


def write_examples_to_jsonls(examples_dict, task_data_path, skip_if_exists=False):
    os.makedirs(task_data_path, exist_ok=True)
    paths_dict = {}
    for phase, example_list in examples_dict.items():
        jsonl_path = os.path.join(task_data_path, f"{phase}.jsonl")
        py_io.write_jsonl(example_list, jsonl_path, skip_if_exists=skip_if_exists)
        paths_dict[phase] = jsonl_path
    return paths_dict


def download_and_unzip(url, extract_location):
    """Downloads and unzips a file, and deletes the zip after"""
    _, file_name = os.path.split(url)
    zip_path = os.path.join(extract_location, file_name)
    download_file(url=url, file_path=zip_path)
    unzip_file(zip_path=zip_path, extract_location=extract_location, delete=True)


def unzip_file(zip_path, extract_location, delete=False):
    """Unzip a file, optionally deleting after"""
    with zipfile.ZipFile(zip_path) as zip_ref:
        zip_ref.extractall(extract_location)
    if delete:
        os.remove(zip_path)


def download_and_untar(url, extract_location):
    """Downloads and untars a file, and deletes the tar after"""
    _, file_name = os.path.split(url)
    tar_path = os.path.join(extract_location, file_name)
    download_file(url, tar_path)
    untar_file(tar_path=tar_path, extract_location=extract_location, delete=True)


def untar_file(tar_path, extract_location, delete=False):
    """Untars a file, optionally deleting after"""
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_location)
    if delete:
        os.remove(tar_path)


def download_file(url, file_path):
    # noinspection PyUnresolvedReferences
    urllib.request.urlretrieve(url, file_path)
