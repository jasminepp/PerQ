"""
Per-rate-point PatchGAN discriminator (Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks",
CVPR 2017) with spectral normalisation (Miyato et al., "Spectral Normalization for Generative Adversarial Networks",
ICLR 2018).
"""
from .utils import *


@dataclass
class RPPatchDiscriminatorConfig:
    in_channels: int = 3
    base_channels: int = 48
    n_layers: int = 3
    num_rp: int = 64
    n_power_iters: int = 1


class _RPSpectralNorm(nn.Module):
    def __init__(self, num_rp, out_channels, in_flat, n_power_iters=1, eps=1.0e-12):
        super().__init__()
        self.num_rp = num_rp
        self.out_channels = out_channels
        self.in_flat = in_flat
        self.n_power_iters = n_power_iters
        self.eps = eps
        self.register_buffer('_u', F.normalize(torch.randn(num_rp, out_channels), dim=-1, eps=eps))

    def forward(self, w_flat):
        assert w_flat.shape == (self.num_rp, self.out_channels, self.in_flat), \
            f'expected w_flat shape [{self.num_rp}, {self.out_channels}, {self.in_flat}], got {tuple(w_flat.shape)}'
        orig_dtype = w_flat.dtype
        with torch.amp.autocast(device_type=w_flat.device.type, enabled=False):
            w_flat_f = w_flat.float()
            if self.training:
                with torch.no_grad():
                    u = self._u
                    for _ in range(self.n_power_iters):
                        v = F.normalize(torch.bmm(w_flat_f.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1),
                                        dim=-1, eps=self.eps)
                        u = F.normalize(torch.bmm(w_flat_f, v.unsqueeze(-1)).squeeze(-1),
                                        dim=-1, eps=self.eps)
                    self._u.copy_(u)
            else:
                u = self._u
                v = F.normalize(torch.bmm(w_flat_f.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1),
                                dim=-1, eps=self.eps)
            sigma = torch.einsum('no,nod,nd->n', u, w_flat_f, v)
            w_norm = w_flat_f / (sigma.view(-1, 1, 1) + self.eps)
        return w_norm.to(orig_dtype)


class _RPConv2d(nn.Module):
    def __init__(self, num_rp, in_channels, out_channels, kernel_size, stride, padding,
                 spectral_norm=False, n_power_iters=50):
        super().__init__()
        self.num_rp = num_rp
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(num_rp, out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(num_rp, out_channels))
        for n in range(num_rp):
            nn.init.kaiming_normal_(self.weight[n], a=0.2, nonlinearity='leaky_relu')
        if spectral_norm:
            self.sn = _RPSpectralNorm(num_rp, out_channels, in_channels * kernel_size * kernel_size,
                                      n_power_iters=n_power_iters)
        else:
            self.sn = None

    def forward(self, x, rp_idx):
        assert x.ndim == 4 and x.shape[1] == self.in_channels, \
            f'expected x shape [B, {self.in_channels}, H, W], got {tuple(x.shape)}'
        assert rp_idx.ndim == 1 and rp_idx.shape[0] == x.shape[0], \
            f'expected rp_idx shape [{x.shape[0]}], got {tuple(rp_idx.shape)}'
        B, _, H, W = x.shape
        if self.sn is not None:
            w_flat = self.weight.view(self.num_rp, self.out_channels, -1)
            w_full = self.sn(w_flat).view_as(self.weight)
        else:
            w_full = self.weight
        w = w_full[rp_idx]
        b = self.bias[rp_idx]
        x = x.reshape(1, B * self.in_channels, H, W)
        w = w.reshape(B * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        b = b.reshape(B * self.out_channels)
        y = F.conv2d(x, w, b, stride=self.stride, padding=self.padding, groups=B)
        return y.reshape(B, self.out_channels, y.shape[-2], y.shape[-1])


class RPPatchDiscriminator(nn.Module):
    """
    Per-rate-point PatchGAN discriminator with spectral normalisation
    """
    def __init__(self, cfg: RPPatchDiscriminatorConfig):
        super().__init__()
        self.cfg = cfg
        self.convs = nn.ModuleList()
        self.convs.append(_RPConv2d(cfg.num_rp, cfg.in_channels, cfg.base_channels, 4, 2, 1,
                                    spectral_norm=True, n_power_iters=cfg.n_power_iters))
        ch = cfg.base_channels
        for _ in range(1, cfg.n_layers):
            ch_next = min(ch * 2, 512)
            self.convs.append(_RPConv2d(cfg.num_rp, ch, ch_next, 4, 2, 1,
                                        spectral_norm=True, n_power_iters=cfg.n_power_iters))
            ch = ch_next
        ch_next = min(ch * 2, 512)
        self.convs.append(_RPConv2d(cfg.num_rp, ch, ch_next, 4, 1, 1,
                                    spectral_norm=True, n_power_iters=cfg.n_power_iters))
        self.final_conv = _RPConv2d(cfg.num_rp, ch_next, 1, 4, 1, 1, spectral_norm=False)

    def forward(self, x, rp_idx):
        for conv in self.convs:
            x = conv(x, rp_idx)
            x = F.leaky_relu(x, 0.2, inplace=True)
        return self.final_conv(x, rp_idx)