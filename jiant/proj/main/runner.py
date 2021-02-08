from typing import Dict, Optional
from dataclasses import dataclass
import logging

import numpy as np
import torch
from tensorboardX import SummaryWriter
import pandas as pd

import jiant.tasks.evaluate as evaluate
import jiant.utils.torch_utils as torch_utils
from jiant.proj.main.components.container_setup import JiantTaskContainer
from jiant.proj.main.modeling.primary import JiantModel, wrap_jiant_forward
from jiant.shared.constants import PHASE
from jiant.shared.runner import (
    complex_backpropagate,
    get_train_dataloader_from_cache,
    get_eval_dataloader_from_cache,
)
from jiant.utils.display import maybe_tqdm
from jiant.utils.python.datastructures import InfiniteYield, ExtendedDataClassMixin
from jiant.utils.logging import regular_log

logger = logging.getLogger(__name__)


@dataclass
class RunnerParameters(ExtendedDataClassMixin):
    local_rank: int
    n_gpu: int
    fp16: bool
    max_grad_norm: float


@dataclass
class TrainState(ExtendedDataClassMixin):
    global_steps: int
    task_steps: Dict[str, int]

    @classmethod
    def from_task_name_list(cls, task_name_list):
        return cls(global_steps=0, task_steps={task_name: 0 for task_name in task_name_list})

    def step(self, task_name):
        self.task_steps[task_name] += 1
        self.global_steps += 1


class JiantRunner:
    def __init__(
        self,
        jiant_task_container: JiantTaskContainer,
        jiant_model: JiantModel,
        optimizer_scheduler,
        device,
        rparams: RunnerParameters,
        log_writer,
        tf_writer: SummaryWriter,
    ):
        self.jiant_task_container = jiant_task_container
        self.jiant_model = jiant_model
        self.optimizer_scheduler = optimizer_scheduler
        self.device = device
        self.rparams = rparams
        self.log_writer = log_writer
        self.tf_writer = tf_writer

        self.model = self.jiant_model

    def get_loss_weights_dict(self, start_position: int = None):
        if start_position is not None:
            raise Exception()
        loss_weights_dict = {}
        for task_name in self.jiant_task_container.task_run_config.train_task_list:
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            logger.info('task="%s": loading loss weights from "%s"',
                        task_name,
                        task_specific_config.train_loss_weights)
            train_batch_size = task_specific_config.train_batch_size
            if task_specific_config.train_loss_weights is not None:
                dataset = pd.read_csv(task_specific_config.train_loss_weights,
                                      sep='\t',
                                      header=None)[0].values
                dataset = torch.Tensor(dataset).to(self.device)
                loss_weights_dict[task_name] = InfiniteYield(
                    torch_utils.DataLoaderWithLength(dataset=dataset, batch_size=train_batch_size)
                )
            else:
                loss_weights_dict[task_name] = None
        return loss_weights_dict

    def run_train(self):
        for _ in self.run_train_context():
            pass

    def run_train_context(self, verbose=True):
        train_dataloader_dict = self.get_train_dataloader_dict()
        loss_weights_dict = {}
        train_state = TrainState.from_task_name_list(
            self.jiant_task_container.task_run_config.train_task_list
        )
        global_train_config = self.jiant_task_container.global_train_config

        losses = []
        for step in maybe_tqdm(
            range(global_train_config.max_steps),
            desc="Training",
            verbose=verbose,
        ):
            regular_log(logger, step, interval=10, tag='train')

            if step == global_train_config.weighted_sampling_start_step:
                train_dataloader_dict = self.get_train_dataloader_dict(do_weighted_sampling=True)
            if step == global_train_config.weighted_loss_start_step:
                loss_weights_dict = self.get_loss_weights_dict()

            loss_per_step = self.run_train_step(
                train_dataloader_dict=train_dataloader_dict, train_state=train_state,
                loss_weights_dict=loss_weights_dict,
            )
            losses.append(loss_per_step)

            if step % 100 == 0:
                logger.info('[train] loss: %f', np.mean(losses))
                self.tf_writer.flush()

            yield train_state

    def resume_train_context(self, train_state, verbose=True):
        train_dataloader_dict = self.get_train_dataloader_dict()
        loss_weights_dict = {}
        start_position = train_state.global_steps
        global_train_config = self.jiant_task_container.global_train_config

        losses = []
        for step in maybe_tqdm(
            range(start_position, global_train_config.max_steps),
            desc="Training",
            initial=start_position,
            total=global_train_config.max_steps,
            verbose=verbose,
        ):
            regular_log(logger, step, interval=10, tag='train')

            if step == global_train_config.weighted_sampling_start_step:
                train_dataloader_dict = self.get_train_dataloader_dict(do_weighted_sampling=True)
            if step >= global_train_config.weighted_loss_start_step:
                loss_weights_dict = self.get_loss_weights_dict()

            loss_per_step = self.run_train_step(
                train_dataloader_dict=train_dataloader_dict, train_state=train_state,
                loss_weights_dict=loss_weights_dict,
            )
            losses.append(loss_per_step)

            if step % 100 == 0:
                logger.info('[train] loss: %f', np.mean(losses))
                self.tf_writer.flush()

            yield train_state

    def run_train_step(self,
                       train_dataloader_dict: dict,
                       loss_weights_dict: dict,
                       train_state: TrainState) -> float:
        self.jiant_model.train()
        task_name, task = self.jiant_task_container.task_sampler.pop()
        task_specific_config = self.jiant_task_container.task_specific_configs[task_name]

        loss_val = 0
        for i in range(task_specific_config.gradient_accumulation_steps):
            batch, batch_metadata = train_dataloader_dict[task_name].pop()

            loss_weights = loss_weights_dict[task_name].pop() if loss_weights_dict.get(task_name, None) is not None else None

            batch = batch.to(self.device)
            model_output = wrap_jiant_forward(
                jiant_model=self.jiant_model, batch=batch, task=task, compute_loss=True,
                loss_weights=loss_weights,
            )

            loss = self.complex_backpropagate(
                loss=model_output.loss,
                gradient_accumulation_steps=task_specific_config.gradient_accumulation_steps,
            )
            loss_val += loss.item()

        self.optimizer_scheduler.step()
        for optimizer in self.optimizer_scheduler.optimizers:
            optimizer.zero_grad()

        train_state.step(task_name=task_name)
        loss_per_step = loss_val / task_specific_config.gradient_accumulation_steps
        self.log_writer.write_entry(
            "loss_train",
            {
                "task": task_name,
                "task_step": train_state.task_steps[task_name],
                "global_step": train_state.global_steps,
                "loss_val": loss_per_step,
            },
        )
        for optimizer in self.optimizer_scheduler.optimizers:
            for i_group, param_group in enumerate(optimizer.param_groups):
                self.tf_writer.add_scalar(f'params-{i_group}/lrate', param_group['lr'], global_step=train_state.global_steps)
        self.tf_writer.add_scalar(f'{task_name}/train-loss', loss_per_step, global_step=train_state.global_steps)
        return loss_per_step

    def run_train_eval(self,
                       task_name_list,
                       global_step: Optional[int] = None,
                       use_subset=None,
                       return_preds=False,
                       return_encoder_output=False,
                       verbose=True):
        dataloader_dict = self.get_train_dataloader_dict(for_eval=True,)
        labels_dict = self.get_train_labels_dict(
            task_name_list=task_name_list, use_subset=use_subset
        )
        return self._run_eval(task_name_list,
                              dataloader_dict,
                              labels_dict,
                              global_step,
                              phase=PHASE.TRAIN,
                              use_subset=use_subset,
                              return_preds=return_preds,
                              return_encoder_output=return_encoder_output,
                              split='train')

    def run_val(self,
                task_name_list,
                global_step: Optional[int] = None,
                use_subset=None,
                return_preds=False,
                return_encoder_output=False,
                verbose=True):
        dataloader_dict = self.get_val_dataloader_dict(
            task_name_list=task_name_list, use_subset=use_subset
        )
        labels_dict = self.get_val_labels_dict(
            task_name_list=task_name_list, use_subset=use_subset
        )
        return self._run_eval(task_name_list,
                              dataloader_dict,
                              labels_dict,
                              global_step=global_step,
                              phase=PHASE.VAL,
                              use_subset=use_subset,
                              return_preds=return_preds,
                              return_encoder_output=return_encoder_output,
                              split='valid')

    def run_test(self, task_name_list, use_subset=None, return_preds=False, verbose=True, return_encoder_output=False):
        dataloader_dict = self.get_test_dataloader_dict()
        labels_dict = self.get_test_labels_dict(
            task_name_list=task_name_list, use_subset=use_subset,
        )

        return self._run_eval(task_name_list, dataloader_dict, labels_dict,
                              phase=PHASE.TEST, use_subset=use_subset,
                              return_preds=return_preds, return_encoder_output=return_encoder_output,
                              split='test')

    def _run_eval(self,
                  task_name_list,
                  dataloader_dict,
                  labels_dict,
                  global_step: Optional[int] = None,
                  phase=None,
                  use_subset=None,
                  return_preds=False,
                  return_encoder_output=False,
                  verbose=True,
                  split='valid'):
        evaluate_dict = {}
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            evaluate_dict[task_name] = run_val(
                val_dataloader=dataloader_dict[task_name],
                val_labels=labels_dict[task_name],
                phase=phase,
                jiant_model=self.jiant_model,
                task=task,
                device=self.device,
                local_rank=self.rparams.local_rank,
                tf_writer=self.tf_writer,
                global_step=global_step,
                return_preds=return_preds,
                return_encoder_output=return_encoder_output,
                verbose=verbose,
                split=split,
            )
        return evaluate_dict

    def get_train_dataloader_dict(self,
                                  for_eval=False,
                                  use_subset=False,
                                  do_weighted_sampling=False):
        # Not currently supported distributed parallel
        train_dataloader_dict = {}
        for task_name in self.jiant_task_container.task_run_config.train_task_list:
            task = self.jiant_task_container.task_dict[task_name]
            train_cache = self.jiant_task_container.task_cache_dict[task_name]["train"]
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            if for_eval:
                train_dataloader_dict[task_name] = get_eval_dataloader_from_cache(
                    eval_cache=train_cache,
                    task=task,
                    eval_batch_size=task_specific_config.eval_batch_size,
                    subset_num=task_specific_config.eval_subset_num if use_subset else None,
                )
            else:
                if do_weighted_sampling:
                    sample_weights_path = task_specific_config.train_sample_weights
                    logger.info('building train loader with sample weights "%s"',
                                task_specific_config.train_sample_weights)
                else:
                    logger.info('building train loader without sample weights')

                    sample_weights_path = None
                train_dataloader_dict[task_name] = InfiniteYield(
                    get_train_dataloader_from_cache(
                        train_cache=train_cache, task=task, train_batch_size=task_specific_config.train_batch_size,
                        sample_weights_path=sample_weights_path,
                        fix_seed_for_weighted_sampler=self.jiant_task_container.global_train_config.fix_seed_for_weighted_sampler,
                    ),
                )
        return train_dataloader_dict

    def _get_eval_dataloader_dict(self, phase, task_name_list, use_subset=False):
        val_dataloader_dict = {}
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            eval_cache = self.jiant_task_container.task_cache_dict[task_name][phase]
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            val_dataloader_dict[task_name] = get_eval_dataloader_from_cache(
                eval_cache=eval_cache,
                task=task,
                eval_batch_size=task_specific_config.eval_batch_size,
                subset_num=task_specific_config.eval_subset_num if use_subset else None,
            )
        return val_dataloader_dict

    def get_val_dataloader_dict(self, task_name_list, use_subset=False):
        return self._get_eval_dataloader_dict(
            phase="val", task_name_list=task_name_list, use_subset=use_subset,
        )

    def get_train_labels_dict(self, task_name_list, use_subset=False):
        return self._get_labels_dict(task_name_list, "train", use_subset=use_subset)

    def get_val_labels_dict(self, task_name_list, use_subset=False):
        return self._get_labels_dict(task_name_list, "val", use_subset=use_subset)

    def get_test_labels_dict(self, task_name_list, use_subset=False):
        return self._get_labels_dict(task_name_list, "test", use_subset=use_subset)

    def _get_labels_dict(self, task_name_list, split: str, use_subset=False):
        labels_dict = {}
        for task_name in task_name_list:
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            labels_cache = self.jiant_task_container.task_cache_dict[task_name][f"{split}_labels"]
            labels = labels_cache.get_all()
            if use_subset:
                labels = labels[: task_specific_config.eval_subset_num]
            labels_dict[task_name] = labels
        return labels_dict

    def get_test_dataloader_dict(self):
        return self._get_eval_dataloader_dict(
            task_name_list=self.jiant_task_container.task_run_config.test_task_list,
            phase=PHASE.TEST,
        )

    def complex_backpropagate(self, loss, gradient_accumulation_steps):
        return complex_backpropagate(
            loss=loss,
            optimizers=self.optimizer_scheduler.optimizers,
            model=self.jiant_model,
            fp16=self.rparams.fp16,
            n_gpu=self.rparams.n_gpu,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=self.rparams.max_grad_norm,
        )

    def get_runner_state(self):
        # TODO: Add fp16  (issue #1186)
        state = {
            "model": torch_utils.get_model_for_saving(self.jiant_model).state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizer_scheduler.optimizers],
        }
        return state

    def load_state(self, runner_state):
        torch_utils.get_model_for_saving(self.jiant_model).load_state_dict(runner_state["model"])
        for optimizer, state_dict in zip(self.optimizer_scheduler.optimizers, runner_state["optimizers"]):
            optimizer.load_state_dict(state_dict)


class CheckpointSaver:
    def __init__(self, metadata, save_path):
        self.metadata = metadata
        self.save_path = save_path

    def save(self, runner_state: dict, metarunner_state: dict):
        to_save = {
            "runner_state": runner_state,
            "metarunner_state": metarunner_state,
            "metadata": self.metadata,
        }
        torch_utils.safe_save(to_save, self.save_path)


def run_val(
    val_dataloader,
    val_labels,
    jiant_model: JiantModel,
    task,
    device,
    local_rank,
    tf_writer: SummaryWriter,
    global_step: Optional[int] = None,
    phase=None,
    return_preds=False,
    return_logits=True,
    return_encoder_output: bool = False,
    verbose=True,
    split='valid',
):
    # Reminder:
    #   val_dataloader contains mostly PyTorch-relevant info
    #   val_labels might contain more details information needed for full evaluation
    has_labels = True  # TODO: データセットにラベルが存在するかどうかを自動判定する．

    if not local_rank == -1:
        return
    jiant_model.eval()
    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()
    total_eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0

    encoder_outputs = []
    for step, (batch, batch_metadata) in enumerate(
        maybe_tqdm(val_dataloader, desc=f"Eval ({task.name}, {str(phase)})", verbose=verbose)
    ):
        regular_log(logger, step, interval=10, tag=split)

        batch = batch.to(device)

        with torch.no_grad():
            model_outputs = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=has_labels, get_encoder_output=return_encoder_output,
            )
            if return_encoder_output:
                model_output, encoder_output = model_outputs
                encoder_outputs.append(encoder_output)
            else:
                model_output = model_outputs
        batch_logits = model_output.logits.detach().cpu().numpy()
        if has_labels:
            batch_loss = model_output.loss.mean().item()
        else:
            batch_loss = 0
        total_eval_loss += batch_loss
        eval_accumulator.update(
            batch_logits=batch_logits,
            batch_loss=batch_loss,
            batch=batch,
            batch_metadata=batch_metadata,
        )

        nb_eval_examples += len(batch)
        nb_eval_steps += 1

    eval_loss = total_eval_loss / nb_eval_steps
    output = {
        "accumulator": eval_accumulator,
    }

    if has_labels:
        tokenizer = (
            jiant_model.tokenizer
            if not torch_utils.is_data_parallel(jiant_model) else jiant_model.module.tokenizer
        )
        metrics = evaluation_scheme.compute_metrics_from_accumulator(
            task=task, accumulator=eval_accumulator, labels=val_labels, tokenizer=tokenizer,
        )

        output.update({
            "loss": eval_loss,
            "metrics": metrics,
        })

        if global_step is not None:
            for metric_name, metric_value in metrics.minor.items():
                tf_writer.add_scalar(f'{split}/{metric_name}', metric_value, global_step=global_step)

    if return_preds:
        output["preds"] = evaluation_scheme.get_preds_from_accumulator(
            task=task, accumulator=eval_accumulator,
        )
        if isinstance(eval_accumulator, evaluate.ConcatenateLogitsAccumulator) and return_logits:
            output["logits"] = eval_accumulator.get_accumulated()
    if return_encoder_output:
        output["encoder_outputs_pooled"] = np.concatenate([
            encoder_output.pooled for encoder_output in encoder_outputs
        ])
        output["encoder_outputs_unpooled"] = np.concatenate([
            encoder_output.unpooled for encoder_output in encoder_outputs
        ])
    if global_step is not None:
        tf_writer.add_scalar(f'{split}/loss', eval_loss, global_step=global_step)

    tf_writer.flush()
    return output


def run_test(
    test_dataloader,
    jiant_model: JiantModel,
    task,
    device,
    local_rank,
    verbose=True,
    return_preds=True,
    return_logits=True,
    return_encoder_output: bool = False,
):
    if not local_rank == -1:
        return
    jiant_model.eval()
    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()

    encoder_outputs = []
    for step, (batch, batch_metadata) in enumerate(
        maybe_tqdm(test_dataloader, desc=f"Eval ({task.name}, Test)", verbose=verbose)
    ):
        regular_log(logger, step, interval=10, tag='test')

        batch = batch.to(device)

        with torch.no_grad():
            model_outputs = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=False, get_encoder_output=return_encoder_output,
            )
            if return_encoder_output:
                model_output, encoder_output = model_outputs
                encoder_outputs.append(encoder_output)
            else:
                model_output = model_outputs
        batch_logits = model_output.logits.detach().cpu().numpy()
        eval_accumulator.update(
            batch_logits=batch_logits, batch_loss=0, batch=batch, batch_metadata=batch_metadata,
        )
    output = {
        "accumulator": eval_accumulator,
    }
    if return_preds:
        output["preds"] = evaluation_scheme.get_preds_from_accumulator(
            task=task, accumulator=eval_accumulator,
        )
        if isinstance(eval_accumulator, evaluate.ConcatenateLogitsAccumulator) and return_logits:
            output["logits"] = eval_accumulator.get_accumulated()
    if return_encoder_output:
        output["encoder_outputs_pooled"] = np.concatenate([
            encoder_output.pooled for encoder_output in encoder_outputs
        ])
        output["encoder_outputs_unpooled"] = np.concatenate([
            encoder_output.unpooled for encoder_output in encoder_outputs
        ])
    return output
