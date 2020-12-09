# coding=utf-8
import logging
import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


logger = logging.getLogger(__name__)


def get_linear_schedule_with_warmup_and_rewarmup(optimizer,
                                                 first_annealing_warmup_steps,
                                                 first_annealing_total_steps,
                                                 second_annealing_warmup_steps,
                                                 second_annealing_total_steps,
                                                 last_epoch=-1):
    w1 = first_annealing_warmup_steps
    t1 = first_annealing_total_steps
    w2 = second_annealing_warmup_steps
    t2 = second_annealing_total_steps

    def lr_lambda(current_step: int):
        c = current_step
        if c < w1:
            return c / max(1, w1)
        elif w1 <= c < t1:
            return max(0.0, (t1 - c) / (max(1, t1 - w1)))
        elif t1 <= c < t1 + w2:
            return (c - t1) / (max(1, w2))
        else:
            return max(
                0.0,
                (t1 + t2 - c) / (max(1, t2 - w2))
            )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
