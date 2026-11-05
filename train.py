"""
PerQ training script (--eval-only true evaluates a checkpoint)
"""
from perq.engine import *
from perq.data import create_datasets


def _parse_args():
    parser = argparse.ArgumentParser(description='PerQ training script', allow_abbrev=False)
    group = parser.add_argument_group('Script parameters')
    group.add_argument('--exp-cfg', type=str, required=True,
                       help='path to experiment config')
    group.add_argument('--model-cfg', type=str, required=True,
                       help='path to model config')
    # Configs overrides
    group.add_argument('--output-dir', type=str, default=None,
                       help='root output directory')
    group.add_argument('--exp-name', type=str, default=None,
                       help='experiment name')
    group.add_argument('--suffix', type=str, default=None,
                       help='suffix appended to experiment name (e.g. "fp16", "resume").')
    group.add_argument('--eval-enable-log', type=str_to_bool, default=None,
                       help='whether to enable logging during evaluation.')
    group.add_argument('--train-cudnn-benchmark', type=str_to_bool, default=None,
                       help='whether to enable torch.backends.cudnn.benchmark during training.')
    group.add_argument('--eval-cudnn-benchmark', type=str_to_bool, default=None,
                       help='whether to enable torch.backends.cudnn.benchmark during evaluation.')
    group.add_argument('--eval-only', type=str_to_bool, default=None,
                       help='whether to only run evaluation.')
    group.add_argument('--pretrain', type=str, default=None,
                       help='path to pretraining checkpoint to load.')
    group.add_argument('--resume', type=str, default=None,
                       help='path to resume training checkpoint to load.')
    group.add_argument('--resume-if-exists', type=str_to_bool, default=None,
                       help='whether to resume training if a checkpoint already exists in the output dir.')
    group.add_argument('--seed', type=int, default=None,
                       help='random seed')

    args = parser.parse_args()

    # Load experiment config
    exp_cfg = MultiStageTrainingConfig(**load_yaml_config(args.exp_cfg))

    # Load model config
    model_cfg = ModelConfig(**load_yaml_config(args.model_cfg))

    # Override args
    for k, v in vars(args).items():
        if v is not None and hasattr(exp_cfg, k):
            setattr(exp_cfg, k, v)

    # CLI override for nested cfg fields
    if args.train_cudnn_benchmark is not None:
        for cfg in exp_cfg.train_cfgs:
            cfg.cudnn_benchmark = args.train_cudnn_benchmark
    if args.eval_cudnn_benchmark is not None:
        exp_cfg.eval_cfg.cudnn_benchmark = args.eval_cudnn_benchmark

    # Set experiment name
    if exp_cfg.exp_name is not None:
        exp_cfg.exp_name += f'_{model_cfg.model_name.lower() if exp_cfg.exp_name.islower() else model_cfg.model_name}'
        exp_cfg.exp_name += f'_{args.suffix}' if args.suffix is not None else ''

    return exp_cfg, model_cfg


def main():
    total_start_time = time.time()

    # Set & store args
    exp_cfg, model_cfg = _parse_args()

    # Set accelerator, logger, output dir
    accelerator, logger, output_dir = get_accelerator_logger(exp_cfg)
    logger.info(f'Output dir: {output_dir}')

    if accelerator.is_main_process:
        # Write config files
        with open(os.path.join(output_dir, 'exp_cfg.yaml'), 'w') as f:
            f.write(yaml.safe_dump(dataclasses.asdict(exp_cfg), default_flow_style=False, indent=4, sort_keys=False))
        with open(os.path.join(output_dir, 'model_cfg.yaml'), 'w') as f:
            f.write(yaml.safe_dump(dataclasses.asdict(model_cfg), default_flow_style=False, indent=4, sort_keys=False))

    accelerator.wait_for_everyone()

    # Set seed
    logger.info(f'Set seed: {exp_cfg.seed}')
    # Per-rank seeds, else every rank draws the same rate points, noise and crops
    accelerate.utils.set_seed(exp_cfg.seed, device_specific=True)

    num_stages = len(exp_cfg.train_cfgs)
    model = None

    # Loop over the training stages (train_cfgs holds one)
    for stage in range(num_stages):
        # Reset losses/metrics/etc. per stage
        eval_losses, eval_metrics, eval_dist_metrics, eval_time = {}, {}, {}, 0.

        logger.info(f'Training stage: {stage + 1}/{num_stages}')

        # Create dataset (eval-only runs build no training side)
        if not exp_cfg.eval_only:
            train_datasets = create_datasets(logger, exp_cfg.train_data_cfgs[0:1],
                                             split='train', force_same_shape=True)
        eval_datasets = create_datasets(logger, exp_cfg.eval_data_cfgs,
                                        split='val', force_same_shape=False)

        # Create tasks
        train_task = None
        if not exp_cfg.eval_only:
            train_task = create_training_task(logger=logger,
                                              root=os.path.join(output_dir, 'train',
                                                                f'rank_{accelerator.process_index}'),
                                              channels=3, bit_depth=8,
                                              rp=exp_cfg.rp, lamb=exp_cfg.lamb,
                                              loss_cfgs=exp_cfg.train_cfgs[stage].loss_cfgs,
                                              metric_cfgs=exp_cfg.train_cfgs[stage].metric_cfgs,
                                              dist_metric_cfgs=[],
                                              rate_loss_warmup_ratio=exp_cfg.train_cfgs[stage].rate_loss_warmup_ratio,
                                              lamb_warmup_ratio=exp_cfg.train_cfgs[stage].lamb_warmup_ratio,
                                              enable_log=False, training=True)
        eval_task = create_training_task(logger=logger,
                                         root=os.path.join(output_dir, 'eval', f'rank_{accelerator.process_index}'),
                                         channels=3, bit_depth=8,
                                         rp=exp_cfg.eval_rp, lamb=exp_cfg.eval_lamb,
                                         loss_cfgs=exp_cfg.eval_cfg.loss_cfgs,
                                         metric_cfgs=exp_cfg.eval_cfg.metric_cfgs,
                                         dist_metric_cfgs=exp_cfg.eval_cfg.dist_metric_cfgs,
                                         enable_log=exp_cfg.eval_enable_log, training=False)

        # Create loader
        train_loaders = []
        if not exp_cfg.eval_only:
            train_batch_size = exp_cfg.train_cfgs[stage].batch_size
            assert train_batch_size % accelerator.num_processes == 0, \
                f"auto_batch_size_scale=True: batch_size ({train_batch_size}) must be divisible by num_processes ({accelerator.num_processes})"
            train_batch_size //= accelerator.num_processes
            logger.info(f'Train batch: cfg={exp_cfg.train_cfgs[stage].batch_size}, per-process={train_batch_size}, '\
                        f'num_processes={accelerator.num_processes}')

            train_loaders = create_training_loader(logger, train_datasets,
                                                   batch_size=train_batch_size,
                                                   shuffle=True,
                                                   num_workers=exp_cfg.workers,
                                                   prefetch_factor=exp_cfg.prefetch_factor,
                                                   pin_memory=exp_cfg.pin_mem)
        eval_loaders = create_training_loader(logger, eval_datasets,
                                              batch_size=exp_cfg.eval_cfg.batch_size,
                                              shuffle=False,
                                              num_workers=exp_cfg.workers,
                                              prefetch_factor=exp_cfg.prefetch_factor,
                                              pin_memory=exp_cfg.pin_mem)

        # Create codec
        if model is None:
            model = create_codec(logger, model_cfg, num_rates=exp_cfg.num_rp)
            model = accelerator.prepare(model)

        # Set stage for the model
        unwrap_model(model).set_stage(stage, num_stages)

        # Optimizer & scheduler
        optimizer, scheduler = None, None
        if not exp_cfg.eval_only:
            stage_opt_sch_cfg = apply_model_opt_overrides(exp_cfg.train_cfgs[stage].opt_sch_cfg, model_cfg,
                                                          logger=logger)
            optimizer, scheduler = create_optimizer_scheduler(logger, stage_opt_sch_cfg, model,
                                                              iters=exp_cfg.train_cfgs[stage].iters,
                                                              warmup_iters=exp_cfg.train_cfgs[stage].warmup_iters)

        # Place per-stage objects on accelerator
        if train_task is not None:
            train_task = accelerator.prepare_model(train_task, evaluation_mode=True)
        eval_task = accelerator.prepare_model(eval_task, evaluation_mode=True)
        if train_loaders:
            train_loaders = accelerator.prepare(*train_loaders) \
                            if len(train_loaders) > 1 else [accelerator.prepare(*train_loaders)]
        eval_loaders = accelerator.prepare(*eval_loaders) \
                       if len(eval_loaders) > 1 else [accelerator.prepare(*eval_loaders)]
        # Drop earlier stages' optimizers: accelerate unscales fp16 gradients once per registered optimizer
        accelerator._optimizers.clear()
        accelerator._schedulers.clear()
        if optimizer is not None:
            optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

        # Count parameters
        num_params = profile_model(logger, unwrap_model(model))

        # Restoring training state per-stage
        iters, ckpt_manager = create_ckpt_and_restore(logger, exp_cfg, accelerator, model,
                                                      output_dir=output_dir, ckpt_name=f'training_stage_{stage}')

        total_stage_iters = exp_cfg.train_cfgs[stage].iters

        # Training loop
        start_train_time = time.perf_counter()
        if not exp_cfg.eval_only:
            if iters < total_stage_iters:
                logger.info(f'Start training stage {stage + 1} for {total_stage_iters - iters} steps.')
            else:
                logger.info(f'Stage {stage + 1} training is already completed. Skip training.')
                train_task.end()
                eval_task.end()
                continue

            if exp_cfg.resume is not None and exp_cfg.eval_on_resume:
                # Evaluation
                start_eval_time = time.perf_counter()
                eval_losses, eval_metrics, eval_dist_metrics = \
                    eval_all(logger, exp_cfg.eval_cfg, accelerator, eval_task,
                             eval_loaders, exp_cfg.eval_data_cfgs, model,
                             log_output=False,
                             output_dir=os.path.join(output_dir, f'eval/{iters}'))
                eval_time = time.perf_counter() - start_eval_time

            while iters < total_stage_iters:
                # Training
                iters, train_losses, train_metrics = \
                    train(logger, exp_cfg.train_cfgs[stage], accelerator, train_task,
                          train_loaders[0], model, optimizer, scheduler,
                          iters)
                train_time = time.perf_counter() - start_train_time

                if do_eval(exp_cfg.train_cfgs[stage], iters):
                    # Evaluation
                    start_eval_time = time.perf_counter()
                    eval_losses, eval_metrics, eval_dist_metrics = \
                        eval_all(logger, exp_cfg.eval_cfg, accelerator, eval_task,
                                 eval_loaders, exp_cfg.eval_data_cfgs, model,
                                 log_output=do_log(exp_cfg.train_cfgs[stage], iters),
                                 output_dir=os.path.join(output_dir, f'eval/{iters}'))
                    eval_time = time.perf_counter() - start_eval_time

                    # Log results and save checkpoint
                    results = {'iters': iters, 'stage': stage + 1,
                               'train_losses': train_losses, 'train_metrics': train_metrics,
                               'eval_losses': eval_losses,
                               'eval_metrics': eval_metrics,
                               'eval_dist_metrics': eval_dist_metrics,
                               'protocol': build_protocol(exp_cfg),
                               'params': num_params,
                               'total_train_time': train_time, 'eval_time': eval_time}

                    write_results(logger, accelerator, results, iters,
                                  output_dir=os.path.join(output_dir, f'eval/{iters}'), mode='a')
                    ckpt_manager.save(iters, {'iters': iters, 'stage': stage, 'train': train_metrics, 'eval': eval_metrics})

        else:
            logger.info(f'Skipping training. Evaluation only.')

            # Evaluation
            start_eval_time = time.perf_counter()
            eval_losses, eval_metrics, eval_dist_metrics = \
                eval_all(logger, exp_cfg.eval_cfg, accelerator, eval_task,
                         eval_loaders, exp_cfg.eval_data_cfgs, model,
                         log_output=True,
                         output_dir=os.path.join(output_dir, f'eval/{iters}'))
            eval_time = time.perf_counter() - start_eval_time

            # Log results and save checkpoint
            results = {'iters': iters, 'stage': stage + 1,
                       'train_losses': {}, 'train_metrics': {},
                       'eval_losses': eval_losses,
                       'eval_metrics': eval_metrics,
                       'eval_dist_metrics': eval_dist_metrics,
                       'protocol': build_protocol(exp_cfg),
                       'params': num_params,
                       'total_train_time': 0, 'eval_time': eval_time}

            write_results(logger, accelerator, results, iters,
                          output_dir=os.path.join(output_dir, f'eval/{iters}'), mode='a')
            ckpt_manager.save(iters, {'iters': iters, 'stage': stage, 'train': {}, 'eval': eval_metrics})

        train_time = time.perf_counter() - start_train_time
        logger.info(f'Stage {stage + 1} completed in: {train_time:.2f}s')

        # Log final results
        results = {'iters': iters, 'stage': stage + 1,
                   'train_losses': {}, 'train_metrics': {},
                   'eval_losses': eval_losses,
                   'eval_metrics': eval_metrics,
                   'eval_dist_metrics': eval_dist_metrics,
                   'protocol': build_protocol(exp_cfg),
                   'params': num_params,
                   'total_train_time': train_time, 'eval_time': eval_time}

        write_results(logger, accelerator, results, model_cfg.model_name,
                      output_dir=os.path.join(output_dir, f'eval'), mode='w')

        # Cleanup between stages
        if train_task is not None:
            train_task.end()
        eval_task.end()
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()

    # Complete training
    total_time = time.time() - total_start_time
    logger.info(f'Training completed in: {total_time:.2f}s')
    logger.info(f'Output are located in: {output_dir}')
    end_training(logger, accelerator)


if __name__ == '__main__':
    main()