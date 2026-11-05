"""
Losses
"""
from perq.utils import *
import pytorch_msssim
import lpips
import DISTS_pytorch


_DTYPE_MAP = {
    'float32':  torch.float32,
    'float16':  torch.float16,
    'bfloat16': torch.bfloat16,
}


def _parse_dtype(dtype):
    """Map a user-friendly dtype string to a torch.dtype. None defaults to float32."""
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in _DTYPE_MAP:
        raise ValueError(f'unknown dtype {dtype!r}; expected one of {list(_DTYPE_MAP)}')
    return _DTYPE_MAP[dtype]


def _check_rgb444(x):
    assert isinstance(x, torch.Tensor) and x.ndim == 5 and x.shape[1] == 3, \
        f'rgb444 input should be a [N,3,T,H,W] tensor, got {type(x).__name__} with shape {x.shape if isinstance(x, torch.Tensor) else "N/A"}'


def _check_shape(x, y):
    assert x.shape == y.shape, 'shape of tensors must be the same!'
    assert x.ndim == y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'


class LossMetricBase(nn.Module):
    def forward(self, x, y, fmt=None):
        return self._compute(x, y)

    def _compute(self, x, y):
        raise NotImplementedError


class LossMetricBaseRGB(nn.Module):
    def forward(self, x, y, fmt=None):
        assert fmt == 'rgb444', f'RGB losses/metrics require rgb444 format, got {fmt}'
        _check_rgb444(x)
        _check_rgb444(y)
        return self._compute(x, y)

    def _compute(self, x, y):
        raise NotImplementedError


def mse(x, y):
    """
    Compute the per-frame MSE loss
    """
    _check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return F.mse_loss(x, y, reduction='none').mean(dim=2)


def psnr(x, y, v_max=1.):
    return 10 * torch.log10((v_max ** 2) / (mse(x, y) + 1e-9))


class MSE(LossMetricBase):
    def _compute(self, x, y):
        return mse(x, y)


class PSNR(LossMetricBase):
    def _compute(self, x, y):
        return psnr(x, y)


def _create_ssim_win(x, size, sigma):
    """
    Create 1-D Gaussian kernel on the target device
    Modified from pytorch-msssim (https://github.com/VainF/pytorch-msssim, MIT License)
    """
    coords = torch.arange(size, dtype=torch.float, device=x.device)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0).repeat([x.shape[1]] + [1] * (len(x.shape) - 1))


# MS-SSIM (Wang et al., "Multiscale Structural Similarity for Image Quality Assessment", Asilomar 2003)
def _ms_ssim(x, y, win_size=11, win_sigma=1.5, v_max=1.):
    _check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    win = _create_ssim_win(x, win_size, win_sigma)
    return pytorch_msssim.ms_ssim(x, y, v_max, win_size=win_size, win_sigma=win_sigma,
                                  win=win, size_average=False).view(N, T)


class MS_SSIM(LossMetricBase):
    def __init__(self, win_size=11, win_sigma=1.5):
        super().__init__()
        self.win_size = win_size
        self.win_sigma = win_sigma

    def _compute(self, x, y):
        return _ms_ssim(x, y, self.win_size, self.win_sigma)


# LPIPS (Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018)
class LPIPS(LossMetricBaseRGB):
    def __init__(self, model='vgg', dtype=None):
        super().__init__()
        self.lpips = lpips.LPIPS(net=model, verbose=False)
        self._autocast_dtype = _parse_dtype(dtype)

    def _compute(self, x, y):
        _check_shape(x, y)
        with torch.amp.autocast(device_type=x.device.type, dtype=self._autocast_dtype):
            return self.lpips(x.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 1),
                              y.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 1), normalize=True) \
                             .view(x.shape[0], x.shape[2])


# E-LPIPS-style loss (Kettunen et al., "E-LPIPS: Robust Perceptual Image Similarity via Random Transformation
# Ensembles", arXiv 2019): LPIPS under one random affine and flip per sample
class E_LPIPS(LPIPS):
    def __init__(self, model='vgg', dtype=None, trans=0.0625, rot_deg=5., scale=0.05, flip_p=0.5):
        super().__init__(model=model, dtype=dtype)
        self.trans = trans
        self.rot_rad = rot_deg * math.pi / 180.
        self.log_scale = math.log(1. + scale)
        self.flip_p = flip_p

    def _sample_theta(self, n, device, dtype):
        u = torch.rand(n, 4, device=device, dtype=dtype) * 2. - 1.
        tx, ty, rt, rs = u.unbind(-1)
        tx = tx * self.trans
        ty = ty * self.trans
        theta = rt * self.rot_rad
        s = torch.exp(rs * self.log_scale)
        flip = 1. - 2. * (torch.rand(n, device=device, dtype=dtype) < self.flip_p).to(dtype)
        sx = s * flip
        sy = s
        c, sn = torch.cos(theta), torch.sin(theta)
        return torch.stack([sx * c, -sy * sn, tx, sx * sn,  sy * c, ty], dim=-1).view(n, 2, 3)

    def _augment(self, x, y):
        n, c, h, w = x.shape
        theta = self._sample_theta(n, x.device, x.dtype)
        grid = F.affine_grid(theta, (n, c, h, w), align_corners=False)
        xy = torch.cat([x, y], dim=0)
        g2 = grid.repeat(2, 1, 1, 1)
        xy = F.grid_sample(xy, g2, mode='bilinear', padding_mode='reflection', align_corners=False)
        return xy[:n], xy[n:]

    def _compute(self, x, y):
        _check_shape(x, y)
        N, C, T, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 1)
        y_flat = y.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 1)
        x_flat, y_flat = self._augment(x_flat, y_flat)
        with torch.amp.autocast(device_type=x.device.type, dtype=self._autocast_dtype):
            return self.lpips(x_flat, y_flat, normalize=True).view(N, T)


# DISTS (Ding et al., "Image Quality Assessment: Unifying Structure and Texture Similarity", TPAMI 2022)
class DISTS(LossMetricBaseRGB):
    def __init__(self, dtype=None):
        super().__init__()
        self.dists = DISTS_pytorch.DISTS()
        self._autocast_dtype = _parse_dtype(dtype)

    def _compute(self, x, y):
        _check_shape(x, y)
        with torch.amp.autocast(device_type=x.device.type, dtype=self._autocast_dtype):
            return self.dists(x.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 1),
                              y.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 1)) \
                             .view(x.shape[0], x.shape[2])


def get_loss_fn(fmt, name, **kwargs):
    assert fmt == 'rgb444', f'fmt must be "rgb444", got {fmt!r}'
    if name == 'mse':
        return MSE()
    elif name == 'e-lpips':
        return E_LPIPS(**kwargs)
    else:
        raise ValueError(f'Unknown loss: {name}')