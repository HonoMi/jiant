import re
from typing import List, Tuple, Optional


def is_task(cv_task_name: str,
            refernce_task_name: str):
    return re.match(f'^{refernce_task_name}__cv-[0-9]+-[0-9]+$', cv_task_name)


def parse_cv_task_name(cv_task_name: str) -> Tuple[str, Optional[int], Optional[int]]:
    if re.match(r'^.*__cv-[0-9]+-[0-9]+$', cv_task_name):
        task_name, n_fold_str, fold_str = re.sub(r'(.*)__cv-([0-9]+)-([0-9]+)',
                                                 r'\g<1>SEP\g<2>SEP\g<3>',
                                                 cv_task_name).split('SEP')
        n_fold = int(n_fold_str)
        fold = int(fold_str)
    else:
        task_name = cv_task_name
        n_fold = None
        fold = None
    return task_name, n_fold, fold


def build_cv_task_name(task_name: str,
                       n_fold: int,
                       fold: int) -> str:
    if n_fold is None:
        return task_name
    else:
        return f'{task_name}__cv-{n_fold}-{fold}'
