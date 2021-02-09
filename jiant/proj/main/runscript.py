import os
import torch
import logging
import traceback

import jiant.proj.main.modeling.model_setup as jiant_model_setup
import jiant.proj.main.runner as jiant_runner
import jiant.proj.main.components.container_setup as container_setup
import jiant.proj.main.metarunner as jiant_metarunner
import jiant.proj.main.components.evaluate as jiant_evaluate
import jiant.shared.initialization as initialization
import jiant.shared.distributed as distributed
import jiant.shared.model_setup as model_setup
import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf

logger = logging.getLogger(__name__)


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    jiant_task_container_config_path = zconf.attr(type=str, required=True)
    output_dir = zconf.attr(type=str, required=True)

    # === Model parameters === #
    model_type = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(default=None, type=str)
    model_tokenizer_path = zconf.attr(default=None, type=str)
    model_load_mode = zconf.attr(default="from_transformers", type=str)

    # === Running Setup === #
    do_train = zconf.attr(action="store_true")
    do_train_eval = zconf.attr(action="store_true")
    do_val = zconf.attr(action="store_true")
    do_test = zconf.attr(action="store_true")

    do_save = zconf.attr(action="store_true")
    do_save_last = zconf.attr(action="store_true")
    do_save_best = zconf.attr(action="store_true")

    load_last_model = zconf.attr(action="store_true")

    write_train_preds = zconf.attr(action="store_true")
    write_val_preds = zconf.attr(action="store_true")
    write_test_preds = zconf.attr(action="store_true")

    write_train_encoder_outputs = zconf.attr(action="store_true")
    write_val_encoder_outputs = zconf.attr(action="store_true")
    write_test_encoder_outputs = zconf.attr(action="store_true")

    eval_every_steps = zconf.attr(type=int, default=0)
    eval_every_epochs = zconf.attr(type=int, default=0)
    save_every_steps = zconf.attr(type=int, default=0)
    save_checkpoint_every_steps = zconf.attr(type=int, default=0)
    no_improvements_for_n_evals = zconf.attr(type=int, default=0)
    keep_checkpoint_when_done = zconf.attr(action="store_true")
    force_overwrite = zconf.attr(action="store_true")

    # Specialized config
    no_cuda = zconf.attr(action="store_true")
    fp16 = zconf.attr(action="store_true")
    fp16_opt_level = zconf.attr(default="O1", type=str)
    local_rank = zconf.attr(default=-1, type=int)
    server_ip = zconf.attr(default="", type=str)
    server_port = zconf.attr(default="", type=str)

    # === Training Learning Parameters === #
    learning_rate = zconf.attr(default=1e-5, type=float)
    adam_epsilon = zconf.attr(default=1e-8, type=float)
    max_grad_norm = zconf.attr(default=1.0, type=float)
    optimizer_type = zconf.attr(default="adam", type=str)
    freeze_encoder = zconf.attr(default=False, type=bool, action="store_true")
    freeze_encoder_when_rewarmup = zconf.attr(default=False, type=bool, action="store_true")
    freeze_top_layer = zconf.attr(default=False, type=bool, action="store_true")
    freeze_top_layer_when_rewarmup = zconf.attr(default=False, type=bool, action="store_true")
    seed = zconf.attr(type=int, default=-1)


@zconf.run_config
class ResumeConfiguration(zconf.RunConfig):
    checkpoint_path = zconf.attr(type=str)


def setup_runner(
    args: RunConfiguration,
    jiant_task_container: container_setup.JiantTaskContainer,
    quick_init_out,
    verbose: bool = True,
) -> jiant_runner.JiantRunner:
    """Setup jiant model, optimizer, and runner, and return runner.

    Args:
        args (RunConfiguration): configuration carrying command line args specifying run params.
        jiant_task_container (container_setup.JiantTaskContainer): task and sampler configs.
        quick_init_out (QuickInitContainer): device (GPU/CPU) and logging configuration.
        verbose: If True, enables printing configuration info (to standard out).

    Returns:
        jiant_runner.JiantRunner

    """
    # TODO document why the distributed.only_first_process() context manager is being used here.
    with distributed.only_first_process(local_rank=args.local_rank):
        # load the model
        jiant_model = jiant_model_setup.setup_jiant_model(
            model_type=args.model_type,
            model_config_path=args.model_config_path,
            tokenizer_path=args.model_tokenizer_path,
            task_dict=jiant_task_container.task_dict,
            taskmodels_config=jiant_task_container.taskmodels_config,
        )
        jiant_model_setup.delegate_load_from_path(
            jiant_model=jiant_model, weights_path=args.model_path, load_mode=args.model_load_mode
        )
        jiant_model.to(quick_init_out.device)

    global_train_config = jiant_task_container.global_train_config
    optimizer_scheduler = model_setup.create_optimizer(
        model=jiant_model,
        learning_rate=args.learning_rate,

        t_total=global_train_config.first_stage_steps,
        warmup_steps=global_train_config.warmup_steps,
        warmup_proportion=None,

        t2_total=global_train_config.second_stage_steps,
        rewarmup_steps=global_train_config.rewarmup_steps,
        rewarmup_proportion=None,

        optimizer_type=args.optimizer_type,
        freeze_encoder=args.freeze_encoder,
        freeze_encoder_when_rewarmup=args.freeze_encoder_when_rewarmup,
        freeze_top_layer=args.freeze_top_layer,
        freeze_top_layer_when_rewarmup=args.freeze_top_layer_when_rewarmup,
        verbose=verbose,
    )
    jiant_model, optimizers = model_setup.raw_special_model_setup(
        model=jiant_model,
        optimizers=optimizer_scheduler.optimizers,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        n_gpu=quick_init_out.n_gpu,
        local_rank=args.local_rank,
    )
    optimizer_scheduler.optimizers = optimizers
    rparams = jiant_runner.RunnerParameters(
        local_rank=args.local_rank,
        n_gpu=quick_init_out.n_gpu,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
    )
    runner = jiant_runner.JiantRunner(
        jiant_task_container=jiant_task_container,
        jiant_model=jiant_model,
        optimizer_scheduler=optimizer_scheduler,
        device=quick_init_out.device,
        rparams=rparams,
        log_writer=quick_init_out.log_writer,
        tf_writer=quick_init_out.tf_writer,
    )
    return runner


def run_loop(args: RunConfiguration, checkpoint=None):
    is_resumed = checkpoint is not None
    quick_init_out = initialization.quick_init(args=args, verbose=True)
    with quick_init_out.log_writer.log_context():
        jiant_task_container = container_setup.create_jiant_task_container_from_json(
            jiant_task_container_config_path=args.jiant_task_container_config_path, verbose=True,
        )
        runner = setup_runner(
            args=args,
            jiant_task_container=jiant_task_container,
            quick_init_out=quick_init_out,
            verbose=True,
        )
        if is_resumed:
            runner.load_state(checkpoint["runner_state"])
            del checkpoint["runner_state"]
        checkpoint_saver = jiant_runner.CheckpointSaver(
            metadata={"args": args.to_dict()},
            save_path=os.path.join(args.output_dir, "checkpoint.p"),
        )
        if args.do_train:
            metarunner = jiant_metarunner.JiantMetarunner(
                runner=runner,
                save_every_steps=args.save_every_steps,
                eval_every_steps=args.eval_every_steps or args.eval_every_epochs * jiant_task_container.global_train_config.steps_per_epoch,
                save_checkpoint_every_steps=args.save_checkpoint_every_steps,
                no_improvements_for_n_evals=args.no_improvements_for_n_evals,
                checkpoint_saver=checkpoint_saver,
                output_dir=args.output_dir,
                verbose=True,
                save_best_model=args.do_save or args.do_save_best,
                save_last_model=args.do_save or args.do_save_last,
                load_best_model=not args.load_last_model,
                log_writer=quick_init_out.log_writer,
                tf_writer=quick_init_out.tf_writer,
            )
            if is_resumed:
                metarunner.load_state(checkpoint["metarunner_state"])
                del checkpoint["metarunner_state"]
            metarunner.run_train_loop()

        if args.do_train_eval or args.write_train_preds or args.write_train_encoder_outputs:
            train_eval_results_dict = runner.run_train_eval(
                task_name_list=runner.jiant_task_container.task_run_config.train_task_list,
                get_preds=args.write_train_preds,
                global_step=None,
                get_encoder_output=args.write_train_encoder_outputs,
            )
            if args.do_train_eval:
                jiant_evaluate.write_train_eval_results(
                    results_dict=train_eval_results_dict,
                    metrics_aggregator=runner.jiant_task_container.metrics_aggregator,
                    output_dir=args.output_dir,
                    verbose=True,
                )
            if args.write_train_preds:
                jiant_evaluate.write_preds(
                    eval_results_dict=train_eval_results_dict,
                    path=os.path.join(args.output_dir, "train_preds.p"),
                )
            if args.write_train_encoder_outputs:
                jiant_evaluate.write_encoder_outputs(
                    eval_results_dict=train_eval_results_dict,
                    output_dir=args.output_dir,
                )
        else:
            assert not args.write_val_preds

        if args.do_val or args.write_val_preds or args.write_val_encoder_outputs:

            val_results_dict = runner.run_val(
                task_name_list=runner.jiant_task_container.task_run_config.val_task_list,
                get_preds=args.write_val_preds,
                get_encoder_output=(True if args.write_val_encoder_outputs else False),
            )
            if args.do_val:
                jiant_evaluate.write_val_results(
                    results_dict=val_results_dict,
                    metrics_aggregator=runner.jiant_task_container.metrics_aggregator,
                    output_dir=args.output_dir,
                    verbose=True,
                )
            if args.write_val_preds:
                jiant_evaluate.write_preds(
                    eval_results_dict=val_results_dict,
                    path=os.path.join(args.output_dir, "val_preds.p"),
                )
            if args.write_val_encoder_outputs:
                jiant_evaluate.write_encoder_outputs(
                    eval_results_dict=val_results_dict,
                    output_dir=args.output_dir,
                )
        else:
            assert not args.write_val_preds

        if args.do_test or args.write_test_preds or args.write_test_encoder_outputs:
            test_results_dict = runner.run_test(
                task_name_list=runner.jiant_task_container.task_run_config.test_task_list,
                get_preds=args.write_test_preds,
                get_encoder_output=(True if args.write_test_encoder_outputs else False),
            )
            if args.do_test:
                jiant_evaluate.write_test_results(
                    results_dict=test_results_dict,
                    metrics_aggregator=runner.jiant_task_container.metrics_aggregator,
                    output_dir=args.output_dir,
                    verbose=True,
                )
            if args.write_test_preds:
                jiant_evaluate.write_preds(
                    eval_results_dict=test_results_dict,
                    path=os.path.join(args.output_dir, "test_preds.p"),
                )
            if args.write_test_encoder_outputs:
                jiant_evaluate.write_encoder_outputs(
                    eval_results_dict=test_results_dict,
                    output_dir=args.output_dir,
                )

    if (
        not args.keep_checkpoint_when_done
        and args.save_checkpoint_every_steps
        and os.path.exists(os.path.join(args.output_dir, "checkpoint.p"))
    ):
        os.remove(os.path.join(args.output_dir, "checkpoint.p"))

    py_io.write_file("DONE", os.path.join(args.output_dir, "done_file"))


def run_resume(args: ResumeConfiguration):
    resume(checkpoint_path=args.checkpoint_path)


def resume(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    args = RunConfiguration.from_dict(checkpoint["metadata"]["args"])
    run_loop(args=args, checkpoint=checkpoint)


def run_with_continue(cl_args):
    run_args = RunConfiguration.default_run_cli(cl_args=cl_args)
    if not run_args.force_overwrite and (
        os.path.exists(os.path.join(run_args.output_dir, "done_file"))
        or os.path.exists(os.path.join(run_args.output_dir, "val_metrics.json"))
    ):
        logger.info("Already Done")
        return
    elif run_args.save_checkpoint_every_steps and os.path.exists(
        os.path.join(run_args.output_dir, "checkpoint.p")
    ):
        logger.info("Resuming")
        resume(os.path.join(run_args.output_dir, "checkpoint.p"))
    else:
        logger.info("Running from start")
        run_loop(args=run_args)


def main():
    mode, cl_args = zconf.get_mode_and_cl_args()
    if mode == "run":
        run_loop(RunConfiguration.default_run_cli(cl_args=cl_args))
    elif mode == "continue":
        run_resume(ResumeConfiguration.default_run_cli(cl_args=cl_args))
    elif mode == "run_with_continue":
        run_with_continue(cl_args=cl_args)
    else:
        raise zconf.ModeLookupError(mode)


if __name__ == "__main__":
    main()
