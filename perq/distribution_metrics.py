"""
Distribution metrics
"""
from perq.utils import *
from perq.losses import *
from perq.losses import _check_rgb444
import torch_fidelity.metric_fid as _mfid
import torch_fidelity.metric_kid as _mkid
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3


_FEATURE_LAYER = '2048'
_FEATURE_DIM = 2048


def _to_uint8_nchw(x):
    _check_rgb444(x)
    x = x.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 1)
    return x.clamp(0., 1.).mul(255.).round().to(torch.uint8)


def _extract_grid(x, patch_size, top, left):
    n, c, h, w = x.shape
    h_avail, w_avail = h - top, w - left
    if h_avail < patch_size or w_avail < patch_size:
        return torch.zeros((0, c, patch_size, patch_size), dtype=x.dtype, device=x.device)
    n_patch_h, n_patch_w = h_avail // patch_size, w_avail // patch_size
    x = x[:, :, top:top + n_patch_h * patch_size, left:left + n_patch_w * patch_size].contiguous()
    return x.view(n, c, n_patch_h, patch_size, n_patch_w, patch_size) \
            .permute(0, 2, 4, 1, 3, 5).contiguous() \
            .view(n * n_patch_h * n_patch_w, c, patch_size, patch_size)


# 256-px patch protocol (Mentzer et al., "High-Fidelity Generative Image Compression", NeurIPS 2020)
def _patchify(x, patch_size, shifted_grid=True):
    assert isinstance(x, torch.Tensor) and x.ndim == 4 and x.shape[1] == 3, \
        f'Input should be a [N,3,H,W] tensor, got {type(x).__name__} with shape {x.shape if isinstance(x, torch.Tensor) else "N/A"}'
    assert x.shape[2] >= patch_size and x.shape[3] >= patch_size, \
        f'Input spatial size {x.shape[2:]} should be >= patch_size {patch_size}'
    grids = [_extract_grid(x, patch_size, 0, 0)]
    if shifted_grid:
        half = patch_size // 2
        grids.append(_extract_grid(x, patch_size, half, half))
    return torch.cat(grids, dim=0)


class DistributionMetricBase(nn.Module):
    feature_name = 'inception-v3'

    def __init__(self, patch_size=256, batch_size=128, shifted_grid=True):
        super().__init__()
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shifted_grid = shifted_grid
        self.feature_extractor = FeatureExtractorInceptionV3(name='inception-v3-compat',
                                                             features_list=[_FEATURE_LAYER])
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)

    @torch.compiler.disable
    def extract(self, x):
        with torch.inference_mode():
            # Convert data
            x_uint8 = _to_uint8_nchw(x)
            if self.patch_size is not None:
                x_uint8 = _patchify(x_uint8, self.patch_size, shifted_grid=self.shifted_grid)

            # Feature extraction
            out = [self.feature_extractor(x_uint8[i:i + self.batch_size])[0] \
                   for i in range(0, x_uint8.shape[0], self.batch_size)]

            # Concat features
            return torch.cat(out, dim=0) if out else \
                   torch.zeros((0, _FEATURE_DIM), device=next(self.feature_extractor.parameters()).device)

    @torch.compiler.disable
    def compute(self, real_features, fake_features):
        raise NotImplementedError


# FID (Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium",
# NeurIPS 2017)
class FID(DistributionMetricBase):
    @torch.compiler.disable
    def compute(self, real_features, fake_features):
        assert real_features.shape[0] > 0 and fake_features.shape[0] > 0, 'Non-empty features are expected'
        real_stats = _mfid.fid_features_to_statistics(real_features.cpu())
        fake_stats = _mfid.fid_features_to_statistics(fake_features.cpu())
        out = _mfid.fid_statistics_to_metric(real_stats, fake_stats, verbose=False)
        return float(out[_mfid.KEY_METRIC_FID])


# KID (Bińkowski et al., "Demystifying MMD GANs", ICLR 2018)
class KID(DistributionMetricBase):
    def __init__(self, patch_size=256, batch_size=128, shifted_grid=True, kid_subsets=100, kid_subset_size=1000):
        super().__init__(patch_size=patch_size, batch_size=batch_size, shifted_grid=shifted_grid)
        self.subsets = kid_subsets
        self.subset_size = kid_subset_size

    @torch.compiler.disable
    def compute(self, real_features, fake_features):
        assert real_features.shape[0] > 0 and fake_features.shape[0] > 0, 'Non-empty features are expected'
        real_features = real_features.cpu()
        fake_features = fake_features.cpu()
        n = min(real_features.shape[0], fake_features.shape[0], self.subset_size)
        out = _mkid.kid_features_to_metric(real_features, fake_features, verbose=False,
                                           kid_subsets=self.subsets, kid_subset_size=n)
        return float(out[_mkid.KEY_METRIC_KID_MEAN])


def get_dist_metric_fn(fmt, name, **kwargs):
    if name == 'fid':
        assert fmt == 'rgb444', f'fid requires rgb444 format, got {fmt}'
        return FID(**kwargs)
    elif name == 'kid':
        assert fmt == 'rgb444', f'kid requires rgb444 format, got {fmt}'
        return KID(**kwargs)
    else:
        raise ValueError(f'Unknown distribution metric: {name}')