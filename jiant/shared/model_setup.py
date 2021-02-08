import transformers
import torch
import logging

from jiant.ext.radam import RAdam
from jiant.shared.model_resolution import ModelArchitectures, resolve_tokenizer_class
from .optimization import get_linear_schedule_with_warmup_and_rewarmup

logger = logging.getLogger(__name__)


def get_tokenizer(model_type, tokenizer_path):
    """Instantiate a tokenizer for a given model type.

    Args:
        model_type (str): model shortcut name.
        tokenizer_path (str): path to tokenizer directory.

    Returns:
        Tokenizer for the given model type.

    """
    model_arch = ModelArchitectures.from_model_type(model_type)
    tokenizer_class = resolve_tokenizer_class(model_type)
    if model_arch in [ModelArchitectures.BERT]:
        if "-cased" in model_type:
            do_lower_case = False
        elif "-uncased" in model_type:
            do_lower_case = True
        else:
            raise RuntimeError(model_type)
    elif model_arch in [
        ModelArchitectures.XLM,
        ModelArchitectures.ROBERTA,
        ModelArchitectures.XLM_ROBERTA,
        ModelArchitectures.BART,
        ModelArchitectures.MBART,
        ModelArchitectures.ELECTRA,
    ]:
        do_lower_case = False
    elif model_arch in [ModelArchitectures.ALBERT]:
        do_lower_case = True
    else:
        raise RuntimeError(str(tokenizer_class))
    tokenizer = tokenizer_class.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)
    return tokenizer


class OptimizerScheduler:
    def __init__(self, optimizers, schedulers):
        super().__init__()
        self.optimizers = optimizers
        self.schedulers = schedulers

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self):
        return {
            "optimizers": [
                optimizer.state_dict()
                for optimizer in self.optimizers
            ],
            "schedulers": [
                scheduler.state_dict()
                for scheduler in self.schedulers
            ],
        }

    def load_state_dict(self, state_dict, strict=True):
        for optimizer, state_dict in zip(self.optimizers, state_dict['optimizer']):
            optimizer.load_state_dict(state_dict, strict=strict)
        for scheduler, state_dict in zip(self.schedulers, state_dict['scheduler']):
            scheduler.load_state_dict(state_dict, strict=strict)
        # self.optimizer.load_state_dict(state_dict["optimizer"], strict=strict)


def create_optimizer(
    model,
    learning_rate,
    t_total,
    warmup_steps,
    warmup_proportion,

    t2_total=None,
    rewarmup_steps=None,
    rewarmup_proportion=None,

    optimizer_epsilon=1e-8,
    optimizer_type="adam",
    freeze_encoder=False,
    freeze_encoder_when_rewarmup=False,
    freeze_top_layer=False,
    freeze_top_layer_when_rewarmup=False,
    verbose=False,
):

    return create_optimizer_from_params(
        model,
        learning_rate=learning_rate,
        t_total=t_total,
        warmup_steps=warmup_steps,
        warmup_proportion=warmup_proportion,
        t2_total=t2_total,
        rewarmup_steps=rewarmup_steps,
        rewarmup_proportion=rewarmup_proportion,
        optimizer_epsilon=optimizer_epsilon,
        optimizer_type=optimizer_type,
        freeze_encoder=freeze_encoder,
        freeze_encoder_when_rewarmup=freeze_encoder_when_rewarmup,
        freeze_top_layer=freeze_top_layer,
        freeze_top_layer_when_rewarmup=freeze_top_layer_when_rewarmup,
        verbose=verbose,
    )


def create_optimizer_from_params(
    model,
    learning_rate,
    t_total,
    warmup_steps,
    warmup_proportion,
    t2_total=None,
    rewarmup_steps=None,
    rewarmup_proportion=None,
    optimizer_epsilon=1e-8,
    optimizer_type="adam",
    freeze_encoder=False,
    freeze_encoder_when_rewarmup=False,
    freeze_top_layer=False,
    freeze_top_layer_when_rewarmup=False,
    verbose=False,
):
    if freeze_encoder and freeze_encoder_when_rewarmup:
        raise ValueError()
    if freeze_top_layer and freeze_top_layer_when_rewarmup:
        raise ValueError()

    # Check
    all_named_parameters = list(model.named_parameters())
    # import pudb; pudb.set_trace()
    # Prepare optimizer
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "adapter.down_project.weight",
        "adapter.up_project.weight",
        "weighted_sum.weights",
    ]
    if verbose:
        logger.info("No optimizer decay for:")
        for n, p in all_named_parameters:
            if any(nd in n for nd in no_decay):
                logger.info(f"  {n}")

    # Group parameters
    encoder = list(model.children())[0]
    model_parameters = model.named_parameters()
    encoder_parameters = [
        (f'encoder.{name}', model_parameters[f'encoder.{name}'])
        for name, _ in encoder.named_parameters()
    ]

    encoder_parameter_names = [name for name, _ in encoder_parameters]
    top_layer_parameters = [(name, val) for name, val in model.named_parameters()
                            if name not in encoder_parameter_names]

    encoder_optimizer_grouped_parameters = []
    top_layer_optimizer_grouped_parameters = []

    for optimizer_grouped_parameters, named_parameters in [(encoder_optimizer_grouped_parameters, encoder_parameters),
                                                           (top_layer_optimizer_grouped_parameters, top_layer_parameters)]:

        used_named_parameters = [
            (n, p) for n, p in named_parameters if p.requires_grad and "weighted_sum.weights" not in n
        ]
        weighted_sum_params = [
            (n, p) for n, p in named_parameters if p.requires_grad and "weighted_sum.weights" in n
        ]
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in used_named_parameters if not any(n.find(nd) >= 0 for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in used_named_parameters if any(n.find(nd) >= 0 for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in weighted_sum_params],
                "weight_decay": 0.0,
                "lr": 0.01
            },
        ])

    # Optimizer
    encoder_lr = learning_rate
    top_layer_lr = learning_rate
    if freeze_encoder:
        encoder_lr = 0.0
    if freeze_top_layer:
        top_layer_lr = 0.0

    if optimizer_type == "adam":
        if verbose:
            logger.info("Using AdamW")
        encoder_optimizer = transformers.AdamW(
            encoder_optimizer_grouped_parameters, lr=encoder_lr, eps=optimizer_epsilon
        )
        top_layer_optimizer = transformers.AdamW(
            top_layer_optimizer_grouped_parameters, lr=top_layer_lr, eps=optimizer_epsilon
        )
    elif optimizer_type == "radam":
        if verbose:
            logger.info("Using RAdam")
        encoder_optimizer = RAdam(encoder_optimizer_grouped_parameters, lr=encoder_lr, eps=optimizer_epsilon)
        top_layer_optimizer = RAdam(top_layer_optimizer_grouped_parameters, lr=top_layer_lr, eps=optimizer_epsilon)
    else:
        raise KeyError(optimizer_type)

    # Scheduler
    warmup_steps = resolve_warmup_steps(
        t_total=t_total, warmup_steps=warmup_steps, warmup_proportion=warmup_proportion,
    )
    if any([t2_total is not None, rewarmup_steps is not None, rewarmup_proportion is not None]):
        second_warmup_steps = resolve_warmup_steps(
            t_total=t2_total, warmup_steps=rewarmup_steps, warmup_proportion=rewarmup_proportion,
        )
        if freeze_encoder_when_rewarmup:
            encoder_second_annealing_lr_scale = 0.0
        if freeze_top_layer_when_rewarmup:
            top_layer_second_annealing_lr_scale = 0.0

        encoder_scheduler = get_linear_schedule_with_warmup_and_rewarmup(
            encoder_optimizer, warmup_steps, t_total, second_warmup_steps, t2_total,
            second_annealing_lr_scale=encoder_second_annealing_lr_scale,
        )
        top_layer_scheduler = get_linear_schedule_with_warmup_and_rewarmup(
            top_layer_optimizer, warmup_steps, t_total, second_warmup_steps, t2_total,
            second_annealing_lr_scale=top_layer_second_annealing_lr_scale,
        )
    else:
        encoder_scheduler = transformers.get_linear_schedule_with_warmup(
            encoder_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
        top_layer_scheduler = transformers.get_linear_schedule_with_warmup(
            top_layer_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

    optimizer_scheduler = OptimizerScheduler(
        optimizers=[encoder_optimizer, top_layer_optimizer],
        schedulers=[encoder_scheduler, top_layer_scheduler],
    )
    return optimizer_scheduler


def resolve_warmup_steps(t_total, warmup_steps, warmup_proportion):
    if warmup_steps is None and warmup_proportion is None:
        raise RuntimeError()
    elif warmup_steps is not None and warmup_proportion is not None:
        raise RuntimeError()
    elif warmup_steps is None and warmup_proportion is not None:
        return warmup_proportion * t_total
    elif warmup_steps is not None and warmup_proportion is None:
        return warmup_steps
    else:
        raise RuntimeError()


def fp16ize(model, optimizers, fp16_opt_level):
    try:
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from apex import amp
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
        )
    model, optimizers = amp.initialize(model, optimizers=optimizers, opt_level=fp16_opt_level)
    return model, optimizers


def parallelize_gpu(model):
    return torch.nn.DataParallel(model)


def parallelize_dist(model, local_rank):
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
    )


def raw_special_model_setup(model, optimizers, fp16, fp16_opt_level, n_gpu, local_rank):
    """Perform setup for special modes (e.g., FP16, DataParallel, and/or DistributedDataParallel.

    Args:
        model (nn.Module): torch model object.
        optimizers: TODO
        fp16 (bool): True to enable FP16 mode.
        fp16_opt_level (str): Apex AMP optimization level default mode identifier.
        n_gpu: number of GPUs.
        local_rank (int): Which GPU the script should use in DistributedDataParallel mode.

    Notes:
        Initialization steps performed in init_cuda_from_args() set n_gpu = 1 when local_rank != -1.

    Returns:
        Model and optimizers with the specified special configuration.

    """
    if fp16:
        model, optimizers = fp16ize(model=model, optimizers=optimizers, fp16_opt_level=fp16_opt_level)
    if n_gpu > 1:
        model = parallelize_gpu(model=model)
    if local_rank != -1:
        model = parallelize_dist(model=model, local_rank=local_rank)
    return model, optimizers


def special_model_setup(
    model_wrapper, optimizer_scheduler, fp16, fp16_opt_level, n_gpu, local_rank
):
    model, optimizers = raw_special_model_setup(
        model=model_wrapper.model,
        optimizers=optimizer_scheduler.optimizers,
        fp16=fp16,
        fp16_opt_level=fp16_opt_level,
        n_gpu=n_gpu,
        local_rank=local_rank,
    )
    model_wrapper.model = model
    optimizer_scheduler.optimizers = optimizers
