"""
Metrics
"""
from perq.utils import *
from perq.losses import *


def get_metric_fn(fmt, name, **kwargs):
    assert fmt == 'rgb444', f'fmt must be "rgb444", got {fmt!r}'
    if name == 'psnr':
        return PSNR()
    elif name == 'ms-ssim':
        return MS_SSIM()
    elif name == 'lpips':
        return LPIPS(**kwargs)
    elif name == 'dists':
        return DISTS(**kwargs)
    else:
        raise ValueError(f'Unknown metric: {name}')