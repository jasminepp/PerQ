"""
Multi-scale per-rate-point PatchGAN discriminator, after Wang et al., "High-Resolution Image Synthesis and Semantic
Manipulation with Conditional GANs", CVPR 2018: one independent PatchGAN per input scale (full / half / quarter). The
gan_module averages the per-scale losses, so the full-resolution score map does not dominate the gradient.
"""
from .utils import *
from .patch_gan import RPPatchDiscriminator, RPPatchDiscriminatorConfig


@dataclass
class MSRPPatchDiscriminatorConfig:
    in_channels: int = 3
    base_channels: int = 48
    n_layers: int = 3
    num_rp: int = 64
    num_scales: int = 3
    n_power_iters: int = 1


class _MultiScaleWrapper(nn.Module):
    def __init__(self, num_scales, build_inner_fn):
        super().__init__()
        assert 1 <= num_scales <= 5, f'num_scales must be in [1, 5], got {num_scales}'
        self.num_scales = num_scales
        self.discs = nn.ModuleList([build_inner_fn() for _ in range(num_scales)])

    def _downsample(self, x, level):
        # avg-pool by 2**level. level=0 -> identity.
        if level == 0:
            return x
        factor = 2 ** level
        return F.avg_pool2d(x, kernel_size=factor, stride=factor)

    def forward(self, x, rp_idx=None):
        outs = []
        for level, disc in enumerate(self.discs):
            xs = self._downsample(x, level)
            outs.append(disc(xs, rp_idx))
        return outs


class MSRPPatchDiscriminator(_MultiScaleWrapper):
    def __init__(self, cfg: MSRPPatchDiscriminatorConfig):
        inner_cfg = RPPatchDiscriminatorConfig(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            n_layers=cfg.n_layers,
            num_rp=cfg.num_rp,
            n_power_iters=cfg.n_power_iters,
        )
        super().__init__(cfg.num_scales, lambda: RPPatchDiscriminator(inner_cfg))
        self.cfg = cfg


def build_model_rp(logger, disc_cfg):
    if isinstance(disc_cfg, dict):
        disc_cfg = MSRPPatchDiscriminatorConfig(**disc_cfg)
    elif disc_cfg is None:
        disc_cfg = MSRPPatchDiscriminatorConfig()
    disc = MSRPPatchDiscriminator(disc_cfg)
    n_params = sum(p.numel() for p in disc.parameters())
    logger.info(f'MSRPPatchDiscriminator: in_channels={disc_cfg.in_channels}  '
                f'base_channels={disc_cfg.base_channels}  n_layers={disc_cfg.n_layers}  '
                f'num_rp={disc_cfg.num_rp}  num_scales={disc_cfg.num_scales}  '
                f'n_power_iters={disc_cfg.n_power_iters}  '
                f'({n_params / 1e6:.2f}M params)')
    return disc