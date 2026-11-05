"""
Configs, optimizer and scheduler, the training and evaluation loops, checkpoints and result files
"""
from perq.utils import *
from perq import tasks
from perq import models

"""
Common settings
"""
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

dynamo.config.verbose = True
dynamo.config.cache_size_limit = 24
dynamo.config.compiled_autograd = True
dynamo.config.optimize_ddp = False


def _generate_lambdas(num_rp, lamb_range):
    assert len(lamb_range) == 2 and lamb_range[0] > 0 and lamb_range[1] >= lamb_range[0], \
        f'lamb_range should be [min, max] with 0 < min <= max, got {lamb_range}'
    if num_rp == 1:
        return [lamb_range[0]]
    return [math.exp(math.log(lamb_range[0]) + i * (math.log(lamb_range[1]) - math.log(lamb_range[0])) / (num_rp - 1)) for i in range(num_rp)]


def _resolve_auto_workers():
    return max(1, min(32, len(os.sched_getaffinity(0)) - 8))


def end_training(logger, accelerator):
    try:
        accelerator.end_training()
    except Exception as e:
        logger.warning(f'accelerator.end_training() raised: {e}')


def _check_nan_and_exit(logger, accelerator, iters_or_epochs, **scalar_losses):
    has_nan = {k: v for k, v in scalar_losses.items()
           if isinstance(v, (int, float)) and not math.isfinite(v)}
    if has_nan:
        logger.error(f'Non-finite training loss at iter/epoch {iters_or_epochs}: {has_nan}. Exiting.')
        accelerator.wait_for_everyone()
        end_training(logger, accelerator)
        sys.exit(1)


def _parse_loss_cfgs(raw):
    if not raw:
        return []
    parsed = []
    for item in raw:
        if isinstance(item, LossConfig):
            parsed.append(item)
        elif isinstance(item, dict):
            parsed.append(LossConfig(**item))
        else:
            raise ValueError(f'Invalid loss_cfgs entry {item!r}; expected dict or LossConfig')
    return parsed

@dataclass
class LossConfig:
    name: str = 'distortion'
    weight: float = 1.0
    lamb_scale: list = field(default_factory=lambda: [True])
    loss_cfgs: list = field(default_factory=lambda: [[1.0, 'rgb444', 'mse']])

    def __post_init__(self):
        assert isinstance(self.lamb_scale, (list, tuple)) and \
               len(self.lamb_scale) in (1, len(self.loss_cfgs)), \
            f'lamb_scale should be a list of 1 or {len(self.loss_cfgs)} bools, got {self.lamb_scale}'
        assert isinstance(self.loss_cfgs, (list, tuple)) and \
               all(isinstance(c, (list, tuple)) and len(c) in (3, 4) for c in self.loss_cfgs), \
            f'loss_cfgs should be a list of [w, fmt, name] or [w, fmt, name, kwargs] entries, got {self.loss_cfgs}'
        assert all(s is True for s in self.lamb_scale), \
            f'lamb_scale must be true for every loss, got {self.lamb_scale!r}'
        assert all(c[1] == 'rgb444' for c in self.loss_cfgs), \
            f'loss fmt must be "rgb444", got {[c[1] for c in self.loss_cfgs]!r}'
        self.weight = float(self.weight)
        self.lamb_scale = [
            bool(s) \
            for s in (self.lamb_scale * len(self.loss_cfgs) if len(self.lamb_scale) == 1 else self.lamb_scale)
        ]
        self.loss_cfgs = [
            (float(c[0]), c[1], c[2], dict(c[3] or {}) if len(c) == 4 else {}) \
            for c in self.loss_cfgs
        ]


@dataclass
class MultiStageTrainingConfig:
    # Rate point config
    num_rp: int = 64
    lamb_range: list = field(default_factory=lambda: [4., 4096.])
    eval_rp: list = field(default_factory=lambda: [0, 9, 18, 27, 36, 45, 54, 63])

    # Output
    output_dir: str = 'outputs'
    exp_name: str = None

    # Logging
    eval_enable_log: bool = True

    # Resuming
    pretrain: str = None
    resume: str = None
    resume_if_exists: bool = False
    eval_on_resume: bool = True

    # Misc
    seed: int = 0
    workers: int = -1
    prefetch_factor: int = -1
    pin_mem: bool = True
    eval_only: bool = False

    # Nested configs
    train_data_cfgs: list = field(default_factory=lambda: [])
    eval_data_cfgs: list = field(default_factory=lambda: [])
    train_cfgs: list = field(default_factory=lambda: [])
    eval_cfg: object = None

    def __post_init__(self):
        # Auto-scale dataloader knobs (-1 → codebase default)
        if self.workers == -1:
            self.workers = _resolve_auto_workers()
        if self.prefetch_factor == -1:
            self.prefetch_factor = 8

        # Train rate points
        self.lamb = _generate_lambdas(self.num_rp, self.lamb_range)

        self.rp = list(range(self.num_rp))
        self.lamb = sorted(self.lamb)

        # Eval rate points
        if self.eval_rp is None:
            self.eval_rp = list(range(self.num_rp))
        assert all(isinstance(rp, int) and 0 <= rp < self.num_rp for rp in self.eval_rp), \
            f'All eval_rp values must be integers in [0, {self.num_rp}), got {self.eval_rp}'

        self.eval_lamb = [self.lamb[r] for r in self.eval_rp]

        # Load nested configs
        assert len(self.train_cfgs) == 1, f'train_cfgs must hold one stage, got {len(self.train_cfgs)}'
        assert len(self.train_data_cfgs) in [1, len(self.train_cfgs)], \
            f'The number of training data configs ({len(self.train_data_cfgs)}) should be 1 or match the number of training configs ({len(self.train_cfgs)})'
        self.train_data_cfgs = [DatasetConfig(**load_yaml_config(cfg)) for cfg in self.train_data_cfgs]
        self.eval_data_cfgs = [DatasetConfig(**load_yaml_config(cfg)) for cfg in self.eval_data_cfgs]
        self.train_cfgs = [TrainingConfig(**load_yaml_config(cfg)) for cfg in self.train_cfgs]
        self.eval_cfg = EvaluationConfig(**load_yaml_config(self.eval_cfg))

        # Validate the loss_cfgs length on parsed train/eval configs
        for i, tc in enumerate(self.train_cfgs):
            assert len(tc.loss_cfgs) == 1, \
                f'train_cfgs[{i}].loss_cfgs must hold one entry, got {len(tc.loss_cfgs)}'
        assert len(self.eval_cfg.loss_cfgs) == 1, \
            f'eval_cfg.loss_cfgs must hold one entry, got {len(self.eval_cfg.loss_cfgs)}'


@dataclass
class TrainingConfig:
    # Training settings
    iters: int = 10000
    warmup_iters: int = 1000
    rate_loss_warmup_ratio: float = 0.0
    lamb_warmup_ratio: float = 0.0
    eval_iters: int = 1000
    log_iters: int = None

    # Batch size
    grad_accum: int = 1
    batch_size: int = 1
    auto_batch_size_scale: bool = True

    # Loss & metric
    loss_cfgs: list = field(default_factory=lambda: [])
    metric_cfgs: list = field(default_factory=lambda: [])

    # Other settings
    max_norm: float = 1.0
    norm_type: float = 2.0

    # Optimizer & scheduler
    opt_sch_cfg: object = None

    # Misc
    cudnn_benchmark: bool = True

    def __post_init__(self):
        self.opt_sch_cfg = OptimizerSchedulerConfig(**(load_yaml_config(self.opt_sch_cfg) \
                                                       if isinstance(self.opt_sch_cfg, str) else self.opt_sch_cfg))
        self.loss_cfgs = _parse_loss_cfgs(self.loss_cfgs)
        # Checkpoints are saved at the evaluations
        assert isinstance(self.eval_iters, int) and self.eval_iters > 0, \
            f'eval_iters must be a positive int, got {self.eval_iters!r}'
        assert self.grad_accum == 1, f'grad_accum must be 1, got {self.grad_accum!r}'
        assert self.auto_batch_size_scale is True, \
            f'auto_batch_size_scale must be true, got {self.auto_batch_size_scale!r}'


@dataclass
class EvaluationConfig:
    # Evaluation
    iters: int = -1

    # Batch size
    batch_size: int = 1

    # Loss & metric
    loss_cfgs: list = field(default_factory=lambda: [])
    metric_cfgs: list = field(default_factory=lambda: [['rgb444', 'psnr'], ['rgb444', 'ms-ssim'],
                                                       ['rgb444', 'lpips', {'model': 'alex', 'dtype': 'float16'}],
                                                       ['rgb444', 'dists', {'dtype': 'float16'}]])

    # Distribution metrics
    dist_metric_cfgs: list = field(default_factory=lambda: [['rgb444', 'fid'], ['rgb444', 'kid']])

    # Misc
    cudnn_benchmark: bool = True

    def __post_init__(self):
        self.loss_cfgs = _parse_loss_cfgs(self.loss_cfgs)
        assert self.iters == -1, f'eval iters must be -1 (the whole dataset), got {self.iters!r}'
        # Real-bitstream coding counts the bits of the whole batch
        assert self.batch_size == 1, f'eval batch_size must be 1, got {self.batch_size!r}'


@dataclass
class DatasetConfig:
    # Dataset
    dataset_name: str = None
    dataset_dir: str = None
    dataset_type: str = 'multi_images'
    video_name: str = None
    video_size: list = field(default_factory=lambda: [-1, -1, -1])
    patch_size: list = field(default_factory=lambda: [1, -1, -1])
    fmt: str = 'png'

    def __post_init__(self):
        assert self.dataset_type == 'multi_images', \
            f'dataset_type must be "multi_images", got {self.dataset_type!r}'
        assert self.video_name is None, f'video_name must be None, got {self.video_name!r}'
        assert tuple(self.video_size) == (-1, -1, -1), f'video_size must be [-1, -1, -1], got {self.video_size!r}'
        assert self.fmt == 'png', f'fmt must be "png", got {self.fmt!r}'


@dataclass
class OptimizerSchedulerConfig:
    opt: str = 'adamw'
    sched: str = 'multistep'
    lr: float = 1e-4
    warmup_lr: float = 1e-6
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    decay_milestones: list = field(default_factory=lambda: [0.5, 0.75, 0.875, 0.9375])

    def __post_init__(self):
        assert self.opt == 'adamw', f'opt must be "adamw", got {self.opt!r}'
        assert self.sched == 'multistep', f'sched must be "multistep", got {self.sched!r}'


_MODEL_OPT_OVERRIDE_FIELDS = (
    'weight_decay',
)


@dataclass
class ModelConfig:
    # Model
    model_name: str = 'DCVC-RT_Intra_PerQ'
    model_type: str = 'dcvc-rt_perq_intra'
    model_cfg: dict = None

    # Overrides opt_sch_cfg.weight_decay of the experiment config (None keeps it)
    weight_decay: float = 1.0e-2

    def __post_init__(self):
        self.model_cfg = load_yaml_config(self.model_cfg) if isinstance(self.model_cfg, str) else self.model_cfg


def apply_model_opt_overrides(opt_sch_cfg, model_cfg, logger=None):
    out = copy.deepcopy(opt_sch_cfg)
    overridden = {}
    for name in _MODEL_OPT_OVERRIDE_FIELDS:
        val = getattr(model_cfg, name, None)
        if val is not None:
            setattr(out, name, val)
            overridden[name] = val
    if overridden and logger is not None:
        logger.info(f'Model-level opt_sch overrides applied: {overridden}')
    return out


def profile_model(logger, model):
    # Compute number of parameters
    num_params = sum(v.numel() for v in model.parameters())
    logger.info(f'Parameters (model): {num_params / 10**6:.2f}M')

    return num_params


"""
Optimizer & scheduler
"""
def get_optimizer_scheduler(logger, opt_sch_cfg, model_or_params, iters, warmup_iters, wd_filter_fn):
    # This is to match the accelerator option 'step_scheduler_with_optimizer=True'
    sched_on_updates = True
    epochs_or_iters = iters
    warmup_epochs_or_iters = warmup_iters
    updates_per_epoch = 1

    # Optimizer & scheduler parameters
    opt = opt_sch_cfg.opt
    sched = opt_sch_cfg.sched

    lr = opt_sch_cfg.lr
    warmup_lr = opt_sch_cfg.warmup_lr
    min_lr = opt_sch_cfg.min_lr
    logger.info(f'Learning rates:')
    logger.info(f'         lr - {lr:.2e}')
    logger.info(f'         warmup_lr - {warmup_lr:.2e}')
    logger.info(f'         min_lr - {min_lr:.2e}')

    weight_decay = opt_sch_cfg.weight_decay
    decay_milestones = [int(epochs_or_iters * m) for m in opt_sch_cfg.decay_milestones]

    # Weight-decay and no-weight-decay groups
    groups = {'default': {'params': [], 'weight_decay': float(opt_sch_cfg.weight_decay)},
              'default_no_wd': {'params': [], 'weight_decay': 0.}}

    for k, v in (model_or_params.named_parameters() if isinstance(model_or_params, nn.Module) else model_or_params):
        if not v.requires_grad:
            continue
        if wd_filter_fn(k, v):
            groups['default']['params'].append(v)
        else:
            groups['default_no_wd']['params'].append(v)

    logger.info(f'Parameter groups:')
    gp_v = groups['default']
    gp_v_no_wd = groups['default_no_wd']
    groups_list = [gp_v, gp_v_no_wd]
    logger.info(f'     Key - default  Weight decay - {gp_v['weight_decay']}  '
                f'Num of params - {len(gp_v['params'])}')
    logger.info(f'     Key - default (No weight decay)  Weight decay - {gp_v_no_wd['weight_decay']}  '
                f'Num of params - {len(gp_v_no_wd['params'])}')

    optimizer = create_optimizer_v2(groups_list,
                                    opt=opt,
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    eps=1e-8,
                                    betas=(0.9, 0.999))
    scheduler, _ = create_scheduler_v2(optimizer,
                                       sched=sched,
                                       num_epochs=epochs_or_iters,
                                       decay_milestones=decay_milestones,
                                       decay_rate=0.5,
                                       min_lr=min_lr,
                                       warmup_lr=warmup_lr,
                                       warmup_epochs=warmup_epochs_or_iters,
                                       step_on_epochs=not sched_on_updates,
                                       updates_per_epoch=updates_per_epoch)
    return optimizer, scheduler


"""
Training/evaluation loops
"""
def train(logger, train_cfg, accelerator, task, loader, model, optimizer, scheduler, iters_or_epochs):
    start_time = time.perf_counter()
    model.train()
    torch.backends.cudnn.benchmark = train_cfg.cudnn_benchmark

    n_updates = iters_or_epochs
    print_every = min(train_cfg.eval_iters, 500)

    accum_loss = []
    accum_metrics = {}
    accum_rate_loss = []
    accum_rate = []
    accum_model_losses = {}
    reduced_loss = 0.
    reduced_metrics = {}
    reduced_rate_loss = 0.
    reduced_rate = 0.
    reduced_model_losses = {}
    n_imgs = 0

    def _forward_step(batch):
        inputs, target, output, loss, metrics, rate, rate_loss, model_losses, _ = task.step(model, batch)
        return inputs, target, output, loss, metrics, rate, rate_loss, model_losses

    def _clip_grad_norm(max_norm):
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm, norm_type=train_cfg.norm_type)

    def _grad_scale_step():
        if train_cfg.max_norm:
            _clip_grad_norm(train_cfg.max_norm)

    def _opt_step():
        optimizer.step()
        optimizer.zero_grad()

    if accelerator.state.dynamo_plugin.backend != accelerate.utils.DynamoBackend.NO:
        _forward_step = torch.compile(_forward_step, **accelerator.state.dynamo_plugin.to_kwargs())
        _grad_scale_step = torch.compile(_grad_scale_step, **accelerator.state.dynamo_plugin.to_kwargs())

    # Initial gradient and optimizer at step 0
    if iters_or_epochs == 0:
        # Get and filter batch
        batch = filter_batch(next(iter(loader)))

        # Perform backward but without stepping
        accelerator.backward(sum(0. * v.sum() for _, v in unwrap_model(model).named_parameters()))
        _grad_scale_step()
        _opt_step()

    while True:
        if hasattr(loader, 'set_epoch'):
            loader.set_epoch(iters_or_epochs // len(loader))

        # Clear buffer and zero grad
        unwrap_model(model).clear_buffer()
        optimizer.zero_grad()

        for i, batch in enumerate(loader, start=0):
            if (i + 1) % print_every == 0 or (i + 1) == 1 or (i + 1) == len(loader):
                task.set_iter(iters_or_epochs, train_cfg.iters)
                unwrap_model(model).set_iter(iters_or_epochs, train_cfg.iters)

            is_last_batch = (iters_or_epochs + 1) >= train_cfg.iters or \
                            (iters_or_epochs + 1) % train_cfg.eval_iters == 0

            # Train step
            # Filter batch
            batch = filter_batch(batch)

            # Step
            inputs, _, outputs, loss, metrics, rate, rate_loss, model_losses = _forward_step(batch)
            total_loss = loss + rate_loss
            total_loss += sum(v for k, v in model_losses.items() if not k.startswith('diag/'))
            accelerator.backward(total_loss)

            # Step optimizer
            _grad_scale_step()
            _opt_step()

            # Update scheduler
            n_updates += 1
            scheduler.step_update(num_updates=n_updates)

            # Accumulate loss and metrics
            update_buffer(accum_loss, loss.detach())
            update_buffer(accum_metrics, {k: v.detach().mean() for k, v in metrics.items()})
            update_buffer(accum_rate_loss, rate_loss.detach())
            update_buffer(accum_rate, rate.detach())
            update_buffer(accum_model_losses, {k: v.detach() for k, v in model_losses.items()})

            n_imgs += inputs['rel_batch_size'] * accelerator.num_processes

            if (i + 1) % print_every == 0 or is_last_batch:
                # Compute averaging loss and metrics globally
                reduced_loss = reduce_buffer(accelerator, accum_loss, to_item=True)
                reduced_metrics = reduce_buffer(accelerator, accum_metrics, to_item=True)
                reduced_rate_loss = reduce_buffer(accelerator, accum_rate_loss, to_item=True)
                reduced_rate = reduce_buffer(accelerator, accum_rate, to_item=True)
                reduced_model_losses = reduce_buffer(accelerator, accum_model_losses, to_item=True)
                _check_nan_and_exit(logger, accelerator, iters_or_epochs,
                                    loss=reduced_loss, rate_loss=reduced_rate_loss,
                                    **reduced_model_losses)

                # Logging loss & metrics
                log_msg = f'Train' + f' - Step [{iters_or_epochs + 1}/{train_cfg.iters}]'
                log_msg += f'    lr: {optimizer.param_groups[0]['lr']:.2e}'
                log_msg += f'    img/s: {n_imgs / (time.perf_counter() - start_time):.2f}'
                log_msg += f'    loss: {reduced_loss + reduced_rate_loss:.4f}'
                log_msg += f'    r_loss: {reduced_rate_loss:.4f}'
                log_msg += f'    d_loss: {reduced_loss:.4f}'
                log_msg += f'    bpp: {reduced_rate:.4f}'
                log_msg += ''.join(f'    model/{k}: {v:.4f}' for k, v in reduced_model_losses.items())
                log_msg += ''.join(f'    {k}: {v:.4f}' for k, v in reduced_metrics.items())
                logger.info(log_msg)
                start_time = time.perf_counter()
                n_imgs = 0

            # Update iters
            iters_or_epochs += 1

            # Terminate training or do evaluation
            if is_last_batch:
                break

        if is_last_batch:
            break

    return iters_or_epochs, {'loss': reduced_loss + reduced_rate_loss}, {'bpp': reduced_rate, **reduced_metrics}


def eval(logger, eval_cfg, accelerator, task, loader, model, log_output=False, output_dir='output'):
    start_time = time.perf_counter()
    model.eval()
    torch.backends.cudnn.benchmark = eval_cfg.cudnn_benchmark

    num_iters = len(loader)
    print_every = min(num_iters, 100)

    all_losses = {}
    all_metrics = {}
    all_dist_metrics = {}

    def _forward_step(batch, rp, lamb):
        inputs, targets, outputs, loss, metrics, rate, rate_loss, model_losses, task_buffers_i = \
            task.step(model, batch, force_rp=rp, force_lamb=lamb)
        return inputs, targets, outputs, loss, metrics, rate, rate_loss, model_losses, task_buffers_i

    if accelerator.state.dynamo_plugin.backend != accelerate.utils.DynamoBackend.NO:
        _forward_step = torch.compile(_forward_step, **accelerator.state.dynamo_plugin.to_kwargs())

    for (rp, lamb) in zip(task.get_rate_points(), task.get_lamb()):
        accum_loss = []
        accum_metrics = {}
        accum_rate_loss = []
        accum_rate = []
        accum_model_losses = {}
        reduced_loss = 0.
        reduced_metrics = {}
        reduced_rate_loss = 0.
        reduced_rate = 0.
        reduced_model_losses = {}
        task_buffers = {}
        dist_metrics = {}
        n_imgs = 0
        rp_tensor = torch.tensor([rp], device=next(iter(model.parameters())).device, dtype=torch.int64)
        lamb_tensor = torch.tensor([lamb], device=next(iter(model.parameters())).device)

        # Clear buffer
        unwrap_model(model).clear_buffer()

        # Load cache if available
        with torch.inference_mode():
            unwrap_model(model).load_cache()

        for i, batch in enumerate(loader, start=0):
            is_last_batch = (i + 1) == len(loader)

            # Eval step
            with torch.inference_mode():
                # Filter batch
                batch = filter_batch(batch)

                # Step
                inputs, _, outputs, loss, metrics, rate, rate_loss, model_losses, task_buffers_i = \
                    _forward_step(batch, rp_tensor, lamb_tensor)

            # Accumulate loss and metrics
            update_buffer(accum_loss, loss.detach(), max=None)
            update_buffer(accum_metrics, {k: v.detach().mean() for k, v in metrics.items()}, max=None)
            update_buffer(accum_rate_loss, rate_loss.detach(), max=None)
            update_buffer(accum_rate, rate.detach(), max=None)
            update_buffer(accum_model_losses, {k: v.detach() for k, v in model_losses.items()}, max=None)

            # Accumulate per-step task buffers
            for key, buf in task_buffers_i.items():
                if key not in task_buffers:
                    task_buffers[key] = []
                task_buffers[key].append(buf)

            n_imgs += inputs['rel_batch_size'] * accelerator.num_processes

            if (i + 1) % print_every == 0 or is_last_batch:
                # Compute averaging loss and metrics globally
                reduced_loss = reduce_buffer(accelerator, accum_loss, to_item=True)
                reduced_metrics = reduce_buffer(accelerator, accum_metrics, to_item=True)
                reduced_rate_loss = reduce_buffer(accelerator, accum_rate_loss, to_item=True)
                reduced_rate = reduce_buffer(accelerator, accum_rate, to_item=True)
                reduced_model_losses = reduce_buffer(accelerator, accum_model_losses, to_item=True)

                # Logging loss & metrics
                log_msg = f'Eval (RP={rp}, Lamb={lamb:.2f})' + f' - Step [{i + 1}/{num_iters}]'
                log_msg += f'    img/s: {n_imgs / (time.perf_counter() - start_time):.2f}'
                log_msg += f'    loss: {reduced_loss + reduced_rate_loss:.4f}'
                log_msg += f'    r_loss: {reduced_rate_loss:.4f}'
                log_msg += f'    d_loss: {reduced_loss:.4f}'
                log_msg += f'    bpp: {reduced_rate:.4f}'
                log_msg += ''.join(f'    model/{k}: {v:.4f}' for k, v in reduced_model_losses.items())
                log_msg += ''.join(f'    {k}: {v:.4f}' for k, v in reduced_metrics.items())
                logger.info(log_msg)
                start_time = time.perf_counter()
                n_imgs = 0

            if is_last_batch:
                # Compute distribution metrics globally
                gathered = gather_buffer(accelerator, task_buffers)
                dist_metrics = task.reduce_buffers(gathered)

                # Logging distribution metrics
                if len(dist_metrics) > 0:
                    log_msg = f'Eval (RP={rp}, Lamb={lamb:.2f})' + f' - Distribution metrics'
                    log_msg += ''.join(f'    {k}: {v:.4f}' for k, v in dist_metrics.items())
                    logger.info(log_msg)
                start_time = time.perf_counter()
                n_imgs = 0

            # Logging outputs
            if log_output:
                task.log_outputs(inputs=inputs, outputs=outputs, metrics=metrics,
                                 rates=rate[None].expand(outputs.shape[0]))

            # Terminate evaluation
            if is_last_batch:
                break

        # Clear cache
        with torch.inference_mode():
            unwrap_model(model).clear_cache()

        # Flush outputs
        if log_output:
            task.flush(dir=os.path.join(output_dir, f'rank_{accelerator.process_index}', f'rp{rp}'))

        # Store loss and metrics for the rate point
        all_losses[rp] = {'loss': reduced_loss + reduced_rate_loss}
        all_metrics[rp] = {'bpp': reduced_rate, **reduced_metrics}
        all_dist_metrics[rp] = dist_metrics

    return all_losses, all_metrics, all_dist_metrics


def eval_all(logger, eval_cfg, accelerator, task, eval_loaders, eval_data_cfgs, model,
             log_output=False, output_dir='output'):
    all_eval_losses = {}
    all_eval_metrics = {}
    all_eval_dist_metrics = {}

    for loader, data_cfg in zip(eval_loaders, eval_data_cfgs):
        name = data_cfg.dataset_name
        logger.info(f'Evaluating dataset: {name}')
        losses, metrics, dist_metrics = eval(logger, eval_cfg, accelerator, task, loader, model,
                                             log_output=log_output,
                                             output_dir=os.path.join(output_dir, name))
        all_eval_losses[name] = losses
        all_eval_metrics[name] = metrics
        all_eval_dist_metrics[name] = dist_metrics

    return all_eval_losses, all_eval_metrics, all_eval_dist_metrics


def do_eval(exp_cfg, iters_or_epochs):
    eval_epochs_or_iters = exp_cfg.eval_iters
    return iters_or_epochs % eval_epochs_or_iters == 0 or \
           iters_or_epochs == exp_cfg.iters


def do_log(exp_cfg, iters_or_epochs):
    log_epochs_or_iters = exp_cfg.log_iters
    if log_epochs_or_iters is None:
        return False
    return (log_epochs_or_iters > 0 and iters_or_epochs % log_epochs_or_iters == 0) or \
           (log_epochs_or_iters == -1 and iters_or_epochs == exp_cfg.iters)


def create_training_loader(logger, train_datasets, batch_size, shuffle, num_workers, prefetch_factor, pin_memory):
    """
    Create the training dataset loader.
    """
    # Create loaders
    loaders = []
    for dataset in train_datasets:
        loader_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': True,
        }
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = prefetch_factor
            loader_kwargs['persistent_workers'] = True
        loaders.append(torch.utils.data.DataLoader(**loader_kwargs))
    return loaders


def filter_batch(batch):
    # Filter batch
    (vidx, vidx_max, idx, idx_max, video_size, patch_size), x = batch
    batch = (vidx, vidx_max, idx, tuple(idx_max[0].tolist()), tuple(video_size[0].tolist()), tuple(patch_size[0].tolist())), x
    return batch


def create_training_task(*args,**kwargs):
    return tasks.create_training_task(*args,**kwargs)


def create_codec(logger, model_cfg, num_rates):
    # Codec
    logger.info(f'Create compression model: {model_cfg.model_name}')
    compress_model = models.create_compress_model(logger, model_cfg, num_rates=num_rates)
    return compress_model


def create_optimizer_scheduler(logger, opt_sch_cfg, model, iters, warmup_iters):
    def _filter_param_for_weight_decay(k, v):
        # Decay the weights (norm scales included), not the biases
        if v.param_type == 'default' and k.endswith('.weight'):
            return True

    logger.info(f'Create optimizer: {opt_sch_cfg.opt}')
    logger.info(f'Create scheduler: {opt_sch_cfg.sched}')
    return get_optimizer_scheduler(logger, opt_sch_cfg, model,
                                   iters=iters, warmup_iters=warmup_iters,
                                   wd_filter_fn=_filter_param_for_weight_decay)


def create_ckpt_and_restore(logger, exp_cfg, accelerator, model, output_dir, ckpt_name):
    # Create checkpoint manager
    ckpt_manager = CheckpointManager(logger, accelerator, os.path.join(output_dir, 'checkpoints', ckpt_name))

    def _log_keys(keys, kind, max_keys=20):
        for k in keys[:max_keys]:
            logger.info(f'Loading pretrained model with {kind} key: {k}')
        if len(keys) > max_keys:
            logger.info(f'... ({len(keys)} {kind} keys in total)')

    # Restore from pretrained model
    if exp_cfg.pretrain:
        assert hasattr(unwrap_model(model), 'load_pretrain'), 'The model should have a load_pretrain method'
        model_path = os.path.expanduser(exp_cfg.pretrain)
        state_dict = torch.load(model_path, map_location=accelerator.device)
        missing_keys, unexpected_keys = unwrap_model(model).load_pretrain(state_dict)
        matched = len(unwrap_model(model).state_dict()) - len(missing_keys)
        logger.info(f'Load pretrain: matched {matched} / missing {len(missing_keys)} / '
                    f'unexpected {len(unexpected_keys)} (src: {model_path})')
        _log_keys(missing_keys, 'missing')
        _log_keys(unexpected_keys, 'unexpected')
        assert matched > 0, f'Load pretrain matched no key. Wrong checkpoint? (src: {model_path})'

    # Restore training state (For resume training)
    epochs_or_iters = 0

    if exp_cfg.resume:
        # Either a run dir or a checkpoint dir given directly
        resume_dir = os.path.expanduser(exp_cfg.resume)
        if not os.path.exists(os.path.join(resume_dir, 'pytorch_model.bin')):
            resume_dir = os.path.join(resume_dir, 'checkpoints', ckpt_name)
        # A missing training state is fatal, not a silent restart from scratch
        assert os.path.exists(resume_dir), f'resume requested but no training state at {resume_dir}'
        logger.info(f'Restore from training state: {resume_dir}')
        epochs_or_iters = ckpt_manager.load(resume_dir)
    else:
        logger.info(f'Start training from scratch.')

    # Per-rank seeds give each rank its own buffers, which DDP does not broadcast here: copy rank 0's parameters
    # and buffers
    if accelerator.distributed_type == accelerate.DistributedType.MULTI_GPU:
        for t in [*model.parameters(), *model.buffers()]:
            torch.distributed.broadcast(t.data, src=0)

    return epochs_or_iters, ckpt_manager


def build_protocol(exp_cfg):
    return {'eval_datasets': {cfg.dataset_name: {'dataset_dir': cfg.dataset_dir,
                                                 'fmt': cfg.fmt}
                              for cfg in exp_cfg.eval_data_cfgs}}


def write_results(logger, accelerator, results, name, output_dir, mode='w'):
    if not accelerator.is_main_process:
        return

    assert mode in ('w', 'a'), 'mode should be "w" or "a"'

    # JSON output
    os.makedirs(output_dir, exist_ok=True)
    out = results
    fpath = os.path.join(output_dir, 'results.json')
    if mode == 'a' and os.path.exists(fpath):
        with open(fpath) as f:
            prev = json.load(f)
        prev.update(out)
        out = prev
    with open(fpath, 'w') as f:
        json.dump({name: out}, f, indent=4)