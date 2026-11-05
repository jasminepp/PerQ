from perq.utils import *
from perq.io_utils import *
from perq.losses import *
from perq.metrics import *
from perq.distribution_metrics import *


class TaskBase(nn.Module):
    def __init__(self, logger, channels, bit_depth, loss_cfgs, metric_cfgs, dist_metric_cfgs=None,
                 channel_scale=None, channel_shift=None, enable_log=True, training=True,
                 rate_loss_warmup_ratio=0., lamb_warmup_ratio=0.):
        super(TaskBase, self).__init__()
        self.logger = logger

        self.channels = channels
        self.bit_depth = bit_depth

        # Channel scale/shift
        assert channel_scale is None or channel_shift is None or \
            (len(channel_scale) == self.channels and len(channel_shift) == self.channels)
        self.channel_scale, self.channel_shift = compute_scale_shift(self.channels, self.bit_depth,
                                                                     channel_scale, channel_shift)

        self.enable_log = enable_log
        self.training = training
        self.metrics_buffer = {}

        # Create loss and metric functions
        self._build_losses_metrics(loss_cfgs, metric_cfgs, dist_metric_cfgs or [])

        # Training progress / warm-up buffers
        self.register_buffer('t', torch.zeros((), dtype=torch.float), persistent=False)
        self.register_buffer('_rate_loss_warmup_ratio',
                             torch.tensor(float(rate_loss_warmup_ratio), dtype=torch.float), persistent=False)
        self.register_buffer('_lamb_warmup_ratio',
                             torch.tensor(float(lamb_warmup_ratio), dtype=torch.float), persistent=False)

    def set_iter(self, iter, num_iters):
        self.t.fill_(iter / num_iters if num_iters > 1 else 0.)

    def _warmup_scale(self, warmup_ratio):
        # Linear ramp t/ratio, clamped to [0, 1]. Returns 1 when ratio == 0 (no warmup).
        return torch.where(warmup_ratio > 0,
                           (self.t / warmup_ratio).clamp(max=1.),
                           torch.ones_like(self.t))

    def _build_losses_metrics(self, loss_cfgs, metric_cfgs, dist_metric_cfgs):
        # Loss cfgs: a list of one LossConfig
        assert isinstance(loss_cfgs, (list, tuple)), \
            f'loss_cfgs should be a list of LossConfig entries, got {type(loss_cfgs)}'
        assert len(loss_cfgs) == 1, f'loss_cfgs must hold one entry, got {len(loss_cfgs)}'
        for cfg in loss_cfgs:
            assert hasattr(cfg, 'name') and hasattr(cfg, 'loss_cfgs'), \
                f'each loss_cfgs entry should have name and loss_cfgs attributes, got {cfg!r}'
            assert isinstance(cfg.loss_cfgs, (list, tuple)) and \
                   all(isinstance(c, (list, tuple)) and len(c) == 4 for c in cfg.loss_cfgs), \
                f'LossConfig.loss_cfgs should be parsed to (weight, fmt, name, kwargs) tuples, got {cfg.loss_cfgs}'
        self.loss_cfgs = loss_cfgs

        # Metric cfgs: list of (fmt, name) or (fmt, name, kwargs) tuples
        assert isinstance(metric_cfgs, (list, tuple)) and \
               all(isinstance(cfg, (list, tuple)) and len(cfg) in (2, 3) for cfg in metric_cfgs), \
            'metric_cfg should be a list of (fmt, name) or (fmt, name, kwargs) tuples'
        self.metric_cfgs = [(fmt, name) for fmt, name, *_ in metric_cfgs]

        # Distribution metric cfgs: list of (fmt, name) or (fmt, name, kwargs)
        assert isinstance(dist_metric_cfgs, (list, tuple)) and \
               all(isinstance(cfg, (list, tuple)) and len(cfg) in (2, 3) for cfg in dist_metric_cfgs), \
            'dist_metric_cfg should be a list of (fmt, name)/(fmt, name, kwargs) tuples'
        self.dist_metric_cfgs = [(fmt, name) for fmt, name, *_ in dist_metric_cfgs]
        assert all(fmt == 'rgb444' for fmt, _ in self.metric_cfgs + self.dist_metric_cfgs), \
            f'metric fmt must be "rgb444", got {[fmt for fmt, _ in self.metric_cfgs + self.dist_metric_cfgs]!r}'

        # Create loss functions, keyed by name
        self.loss_fns = nn.ModuleDict()
        for cfg in self.loss_cfgs:
            for _, fmt, name, kwargs in cfg.loss_cfgs:
                if name not in self.loss_fns:
                    self.loss_fns[name] = get_loss_fn(fmt, name, **kwargs)
        self.metric_fns = nn.ModuleDict({
            name: get_metric_fn(fmt, name, **((cfg[2] or {}) if len(cfg) == 3 else {}))
            for cfg, (fmt, name) in zip(metric_cfgs, self.metric_cfgs)
        })
        self.dist_metric_fns = nn.ModuleDict({
            name: get_dist_metric_fn(fmt, name, **((cfg[2] or {}) if len(cfg) == 3 else {}))
            for cfg, (fmt, name) in zip(dist_metric_cfgs, self.dist_metric_cfgs)
        })

        # Sanity-check feature sharing - metrics that share features should use the same patch_size
        feature_keys = {}
        for fmt, name in self.dist_metric_cfgs:
            metric = self.dist_metric_fns[name]
            key = (metric.feature_name, fmt)
            if key in feature_keys:
                assert feature_keys[key].patch_size == metric.patch_size, \
                    f'distribution metrics sharing {key} must use the same patch_size'
            else:
                feature_keys[key] = metric

    def get_rate_points(self):
        raise NotImplementedError("This method should be implemented in the subclass.")

    def get_lamb(self):
        return self.get_lamb_tensor().tolist()

    def get_lamb_tensor(self):
        lamb = self._lamb.min() + (self._lamb - self._lamb.min()) * self._warmup_scale(self._lamb_warmup_ratio)
        return lamb

    def prepare_color_spaces(self, x, y):
        # The codec, losses and metrics all work in rgb444
        return {'rgb444': x}, {'rgb444': y}

    def compute_d_loss(self, x_fmts, y_fmts, lamb):
        loss = 0.

        for cfg in self.loss_cfgs:
            for weight, fmt, name, _kwargs in cfg.loss_cfgs:
                loss_i = self.loss_fns[name](x_fmts[fmt], y_fmts[fmt], fmt=fmt)
                weighted = lamb.view(-1, 1) * cfg.weight * weight * loss_i
                loss += weighted.sum() / (lamb.shape[0] * loss_i.shape[1])
        return loss

    def compute_r_loss(self, r):
        return self._warmup_scale(self._rate_loss_warmup_ratio).to(r.dtype) * r

    def compute_metrics(self, x_fmts, y_fmts):
        with torch.no_grad():
            metrics = {}
            x_fmts_clamped = {fmt: x_fmt.clamp(0., 1.) for fmt, x_fmt in x_fmts.items()}
            y_fmts_clamped = {fmt: y_fmt.clamp(0., 1.) for fmt, y_fmt in y_fmts.items()}
            for fmt, name in self.metric_cfgs:
                display_name = f'{name} ({fmt})'
                metrics[display_name] = self.metric_fns[name](x_fmts_clamped[fmt], y_fmts_clamped[fmt], fmt=fmt)
            return metrics

    def compute_buffers(self, x_fmts, y_fmts):
        with torch.no_grad():
            buffers = {}
            x_fmts_clamped = {fmt: x_fmt.clamp(0., 1.) for fmt, x_fmt in x_fmts.items()}
            y_fmts_clamped = {fmt: y_fmt.clamp(0., 1.) for fmt, y_fmt in y_fmts.items()}
            for fmt, name in self.dist_metric_cfgs:
                metric = self.dist_metric_fns[name]
                feature_key = f'{metric.feature_name} ({fmt})'
                if feature_key not in buffers:
                    buffers[feature_key] = {
                        'real': metric.extract(y_fmts_clamped[fmt]),
                        'fake': metric.extract(x_fmts_clamped[fmt]),
                    }
            return buffers

    def reduce_buffers(self, buffers):
        out = {}
        for fmt, name in self.dist_metric_cfgs:
            metric = self.dist_metric_fns[name]
            feature_key = f'{metric.feature_name} ({fmt})'
            buf = buffers[feature_key]
            out[f'{name} ({fmt})'] = metric.compute(buf['real'], buf['fake'])
        return out

    def step(self, model, batch, force_rp=None, force_lamb=None):
        raise NotImplementedError

    def end(self):
        pass


class TrainingTask(TaskBase):
    def __init__(self, logger, root, channels, bit_depth,
                 rp, lamb, loss_cfgs, metric_cfgs, dist_metric_cfgs=None,
                 channel_scale=None, channel_shift=None, rate_loss_warmup_ratio=0., lamb_warmup_ratio=0.,
                 enable_log=True, training=True):
        super(TrainingTask, self).__init__(logger=logger,
                                           channels=channels, bit_depth=bit_depth,
                                           loss_cfgs=loss_cfgs, metric_cfgs=metric_cfgs,
                                           dist_metric_cfgs=dist_metric_cfgs,
                                           channel_scale=channel_scale, channel_shift=channel_shift,
                                           rate_loss_warmup_ratio=rate_loss_warmup_ratio,
                                           lamb_warmup_ratio=lamb_warmup_ratio,
                                           enable_log=enable_log, training=training)
        self.root = root

        # Rate point settings
        assert isinstance(rp, (list, tuple)) and isinstance(lamb, (list, tuple)) and len(rp) == len(lamb), \
            f'rp and lamb should be lists/tuples with equal length, got {len(rp)} and {len(lamb)}'
        self.rp = list(rp)
        self.lamb = list(lamb)
        self.register_buffer('_lamb', torch.tensor(self.lamb, dtype=torch.float), persistent=False)

        logger.info(f'TrainingTask:')
        logger.info(f'     Root: {self.root}')
        logger.info(f'     Channels: {self.channels}    Bit depth: {self.bit_depth}')
        logger.info(f'     Rate points: {self.rp}')
        logger.info(f'     Lamb: {[f"{l:.2f}" for l in self.lamb]}')
        logger.info(f'     Losses: {self.loss_cfgs}    Metrics: {self.metric_cfgs}')
        logger.info(f'     Enable log: {self.enable_log}')
        logger.info(f'     Training: {self.training}')

        # Create output directory
        os.makedirs(self.root, exist_ok=True)

    def get_rate_points(self):
        return list(self.rp)

    def parse_batch(self, batch, force_rp=None, force_lamb=None):
        """
        Parse the input and output batch during training/evaluation step
        """
        (vidx, vidx_max, idx, idx_max, video_size, patch_size), x = batch
        assert vidx.ndim == 1, 'vidx should have 1 dimension with shape [N]'
        assert idx.ndim == 2, 'idx should have 2 dimensions with shape [N, 3]'
        assert len(idx_max) == 3, 'idx_max should be a tuple with length 3'
        assert len(video_size) == 3, 'video_size should be a tuple with length 3'
        assert len(patch_size) == 3, 'patch_size should be a tuple with length 3'
        assert x.ndim == 5,  'x should have 5 dimensions with shape [N, C, T, H, W]'

        if force_rp is None:
            # Randomly sample from available rate points
            rp_idx = torch.randint(0, len(self.rp), (x.shape[0],), device=vidx.device)
            rp = torch.tensor(self.rp, device=vidx.device, dtype=torch.int64)[rp_idx]
            lamb = self.get_lamb_tensor()[rp_idx]
        else:
            rp = force_rp.expand(x.shape[0])
            lamb = force_lamb.expand(x.shape[0])
        assert rp.shape == (x.shape[0],), 'rp should have shape [N]'
        assert lamb.shape == (x.shape[0],), 'lamb should have shape [N]'

        inputs = {
            'rp': rp,
            'lamb': lamb,
            'batch_size': torch.tensor(x.shape[0], dtype=torch.int32, device=vidx.device),
            'rel_batch_size': x.shape[0] * x.shape[2] * math.prod(patch_size[1:]) / math.prod(video_size[1:]),
            'vidx': vidx,
            'vidx_max': vidx_max,
            'idx': idx,
            'idx_max': idx_max,
            'video_size': video_size,
            'patch_size': patch_size,
            'channels': x.shape[1],
            'x': x,
        }

        return inputs, x

    def parse_output(self, output):
        """
        Parse the output from the model during training/evaluation step
        """
        return output.contiguous()

    def step(self, model, batch, force_rp=None, force_lamb=None):
        """
        Step function for training/evaluation
        """
        inputs, target = self.parse_batch(batch, force_rp=force_rp, force_lamb=force_lamb)
        output, rate, rate_loss, model_losses = model(inputs, compute_outputs=True, compute_rates=True)
        unwrap_model(model).clear_buffer()
        output = self.parse_output(output)
        return self.step_tail(inputs, target, output, rate, rate_loss, model_losses)

    def step_tail(self, inputs, target, output, rate, rate_loss, model_losses):
        """
        Tail function for processing the outputs, computing losses and metrics, and returning results
        """
        x_fmts, y_fmts = self.prepare_color_spaces(output.float(), target.float())
        loss = self.compute_d_loss(x_fmts, y_fmts, inputs['lamb'])
        metrics = self.compute_metrics(x_fmts, y_fmts)
        buffers = self.compute_buffers(x_fmts, y_fmts)
        num_pixels = inputs['rel_batch_size'] * math.prod(inputs['video_size'][1:])
        rate = rate.sum() / num_pixels
        rate_loss = self.compute_r_loss(rate_loss.sum() / num_pixels)
        return inputs, target, output, loss, metrics, rate, rate_loss, model_losses, buffers

    def log_outputs(self, inputs, outputs, metrics, rates):
        """
        Log the outputs
        """
        if self.enable_log:
            N, _, T, _, _ = outputs.shape
            vidx = inputs['vidx']
            idx = inputs['idx']
            outputs = outputs.detach().clamp(0., 1.)
            metrics = {k: v.detach() for k, v in metrics.items()}

            scale = self.channel_scale.to(outputs.device)
            shift = self.channel_shift.to(outputs.device)

            # Loop over all samples
            for n in range(N):
                for t in range(T):
                    # Save the patch
                    assert self.bit_depth == 8, 'Only 8-bit depth is supported for logging'
                    torchvision.io.write_png(
                        ((outputs[n, :, t] - shift.view(self.channels, 1, 1)) / \
                         scale.view(self.channels, 1, 1)).round().to(torch.uint8).cpu(),
                        os.path.join(self.root, f'{vidx[n].cpu().numpy()}_{t}.png')
                    )

                # Save the average metrics for each patch
                idx_str = '(' + str(vidx[n].item()) + ', (' + ' '.join([str(i) for i in idx[n].tolist()]) + '))'
                self.metrics_buffer[idx_str] = \
                    ','.join([idx_str] + [f'{k}: {v[n].mean().item():.4f}' for k, v in metrics.items()] +
                             [f'rate: {rates[n].item():.4f}'])

    def flush(self, dir):
        """
        Flush the outputs to the disk
        """
        if self.enable_log:
            # Save the reconstructions
            os.makedirs(dir, exist_ok=True)
            for file in os.listdir(self.root):
                os.rename(os.path.join(self.root, file), os.path.join(dir, file))

            # Save metrics
            with open(os.path.join(dir, 'metrics.txt'), 'w') as f:
                f.write('\n'.join(self.metrics_buffer.values()))
            self.metrics_buffer = {}

    def end(self):
        shutil.rmtree(self.root)


def create_training_task(**kwargs):
    # Create task
    return TrainingTask(**kwargs)