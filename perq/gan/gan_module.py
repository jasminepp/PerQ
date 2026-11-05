"""
Hinge GAN loss with a fixed weight, enabled at start_iter and warmed up linearly
"""
from .utils import *
from .utils import _get_g_loss_fn, _get_d_loss_fn


def _apply_loss(loss_fn, scores, *args):
    return torch.stack([loss_fn(s, *args) for s in scores]).mean()


def _apply_d_loss(d_loss_fn, real_scores, fake_scores, mask):
    return torch.stack([d_loss_fn(r, f, mask) for r, f in zip(real_scores, fake_scores)]).mean()


def _flatten_scores(scores):
    return torch.cat([s.detach().flatten() for s in scores])


@dataclass
class GANConfig:
    disc_type: str = 'ms_rp_patch_gan'
    disc_cfg: dict = None
    weight_max: float = 0.1
    weight_type: str = 'fixed'
    start_iter: int = 20000
    warmup_iters: int = 20000
    g_loss_type: str = 'hinge'
    d_loss_type: str = 'hinge'
    enable_diag: bool = True

    def __post_init__(self):
        assert self.disc_type == 'ms_rp_patch_gan', f'disc_type must be "ms_rp_patch_gan", got {self.disc_type!r}'
        assert isinstance(self.weight_max, (int, float)) and not isinstance(self.weight_max, bool), \
            f'weight_max must be a number, got {self.weight_max!r}'
        assert self.weight_type == 'fixed', f'weight_type must be "fixed", got {self.weight_type!r}'
        assert self.g_loss_type == 'hinge', f'g_loss_type must be "hinge", got {self.g_loss_type!r}'
        assert self.d_loss_type == 'hinge', f'd_loss_type must be "hinge", got {self.d_loss_type!r}'
        assert self.enable_diag is True, f'enable_diag must be true, got {self.enable_diag!r}'
        self.weight_max = float(self.weight_max)


class GANModule(nn.Module):
    def __init__(self, logger, cfg: GANConfig):
        super().__init__()
        self.cfg = cfg
        from . import create_discriminator   # local import to avoid circular import at module load
        self.discriminator = create_discriminator(logger, cfg)
        self.g_loss_fn = _get_g_loss_fn(cfg.g_loss_type)
        self.d_loss_fn = _get_d_loss_fn(cfg.d_loss_type)

        self._weight_max = 0.0
        self.register_buffer('_current_stage_iter',
                             torch.tensor(0, dtype=torch.long),
                             persistent=False)
        self._enabled = False

    def set_stage(self, stage, num_stages):
        self._current_stage_iter.zero_()
        self._weight_max = self.cfg.weight_max
        # Re-evaluate the enable flag for the new stage (start_iter=0 enables at iter 0).
        self._enabled = int(self.cfg.start_iter) <= 0

    def set_iter(self, iter, num_iters):
        self._current_stage_iter.fill_(int(iter))
        if not self._enabled and int(self._current_stage_iter.item()) >= self.cfg.start_iter:
            self._enabled = True

    def is_active(self):
        return self.training and self._weight_max > 0

    def compute_losses(self, x_real, x_fake, rp_idx=None):
        if not self.is_active():
            return {}

        # Handle 5D → 4D (codec output is [B, C, 1, H, W])
        if x_real.ndim == 5:
            x_real = x_real.squeeze(2)
        if x_fake.ndim == 5:
            x_fake = x_fake.squeeze(2)

        # Linear warmup: ramps gan_g / gan_d contribution from 0 (pre-start_iter) to full.
        # Always >= 0 so disc params are kept in the autograd graph at all times.
        iters_since_start = (self._current_stage_iter - int(self.cfg.start_iter) + 1).to(x_fake.dtype)
        warmup_iters = int(self.cfg.warmup_iters)
        if warmup_iters > 0:
            warmup_factor = (iters_since_start / warmup_iters).clamp(min=0.0, max=1.0)
        else:
            warmup_factor = (iters_since_start >= 1.0).to(x_fake.dtype)
        # All-ones sample mask: the hinge losses normalise by mask.sum() * numel in x_fake's dtype
        mask = torch.ones(x_fake.shape[0], device=x_fake.device, dtype=x_fake.dtype)

        # Pre-start_iter shortcut: skip the disc forward entirely (which can be heavy + can
        # produce NaN under fp16 with extreme inputs). Emit a zero loss that touches all disc
        # params so DDP still sees them as used. Generator params are kept in the graph by the
        # surrounding MSE/LPIPS losses.
        # Uses self._enabled (a Python bool flipped by set_iter when iter >= start_iter) to
        # avoid forcing a CPU sync via .item() — that would break torch.compile graphs.
        if not self._enabled:
            zero_disc = sum(p.sum() for p in self.discriminator.parameters() if p.requires_grad)
            zero_disc = zero_disc * 0.0
            out = {'gan_g': zero_disc, 'gan_d': zero_disc}
            if self.cfg.enable_diag:
                z = torch.zeros((), device=x_fake.device, dtype=x_fake.dtype)
                out.update({
                    'diag/gan_d_real': z, 'diag/gan_d_fake': z,
                    'diag/gan_d_acc': z + 0.5,
                    'diag/gan_d_weight': torch.tensor(self._weight_max, device=x_fake.device, dtype=x_fake.dtype),
                    'diag/gan_warmup': z,
                })
            return out

        # G-loss: freeze D so g_loss gradient only reaches the generator
        for p in self.discriminator.parameters():
            p.requires_grad_(False)
        fake_scores_g = self.discriminator(x_fake, rp_idx)
        g_loss = _apply_loss(self.g_loss_fn, fake_scores_g, mask)
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        gan_w = torch.tensor(self._weight_max, device=x_fake.device, dtype=x_fake.dtype)

        # D-loss: detach fake so d_loss gradient only reaches the discriminator
        fake_scores_d = self.discriminator(x_fake.detach(), rp_idx)
        real_scores = self.discriminator(x_real, rp_idx)
        d_loss = _apply_d_loss(self.d_loss_fn, real_scores, fake_scores_d, mask)

        out = {
            'gan_g': warmup_factor * gan_w * g_loss,
            'gan_d': warmup_factor * d_loss,
        }
        if self.cfg.enable_diag:
            with torch.no_grad():
                # Aggregate diag across scales: flatten all per-scale score maps into one vector.
                real_flat = _flatten_scores(real_scores)
                fake_flat = _flatten_scores(fake_scores_d)
                d_real_mean = real_flat.mean()
                d_fake_mean = fake_flat.mean()
                d_acc = 0.5 * ((real_flat > 0).float().mean() + (fake_flat < 0).float().mean())
                out.update({
                    'diag/gan_d_real': d_real_mean,
                    'diag/gan_d_fake': d_fake_mean,
                    'diag/gan_d_acc': d_acc,
                    'diag/gan_d_weight': gan_w.detach(),
                    'diag/gan_warmup': warmup_factor.detach(),
                })
        return out