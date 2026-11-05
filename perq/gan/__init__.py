from . import gan_module
from . import patch_gan
from . import multi_scale_patch_gan


def create_discriminator(logger, gan_cfg):
    """Build the discriminator named by gan_cfg.disc_type."""
    if gan_cfg.disc_type == 'ms_rp_patch_gan':
        return multi_scale_patch_gan.build_model_rp(logger, gan_cfg.disc_cfg)
    else:
        raise ValueError(f'Unknown disc_type: {gan_cfg.disc_type}')