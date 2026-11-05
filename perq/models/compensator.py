"""
PerQ compensator: a decoder-side generator of the latent quantization error, bounded to the quantization cell.

Interface:
    loss  = compensator.compute_loss(y_hat, z_hat, q_dec, delta_target)
    delta = compensator.sample(y_hat, z_hat, q_dec, num_steps=1, noise=None)

  - y_hat        : [B, latent_ch, yH,   yW  ] quantized latent in SYMBOL space (pre-q_dec)
  - z_hat        : [B, cond_ch,   yH/4, yW/4] hyper-prior latent (compensator upsamples 4×)
  - q_dec        : [B, latent_ch, yH,   yW  ] combined quantization step (= q_dec_local · q_dec_bott)
  - delta_target : [B, latent_ch, yH,   yW  ] training target = y_scaled - y_hat (symbol-space residual)

FlowMatchingCompensator: conditional flow matching with a ConvNeXt trunk, Euler sampling.
"""

from perq.utils import *


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding for t in [0, 1]; max_period=10 gives frequencies in [0.1, 1]."""
    def __init__(self, dim, max_period=10):
        super().__init__()
        assert dim % 2 == 0, f'SinusoidalPosEmb dim must be even, got {dim}'
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freq = torch.exp(-math.log(self.max_period) * torch.arange(half, device=device) / half)
        args = t[:, None] * freq[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (Liu et al., "A ConvNet for the 2020s", CVPR 2022) with adaLN-zero (Peebles and Xie, "Scalable
    Diffusion Models with Transformers", ICCV 2023): shift, scale, gate from t_emb, gate zero-init so the block starts
    as shortcut(x). Requires t_emb.shape[-1] == out_ch."""

    def __init__(self, in_ch, out_ch, expansion=2):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        # 5x5 dwconv: 10 stacked blocks give RF=41, enough for the 16x16 training latent without being too global
        # for 64x64 inference
        self.dwconv = nn.Conv2d(in_ch, in_ch, 5, padding=2, groups=in_ch)
        # Non-affine: adaLN supplies scale/shift
        self.norm = LayerNorm2d(in_ch)
        self.pwconv1 = nn.Linear(in_ch, in_ch * expansion)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(in_ch * expansion, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.adaLN = nn.Linear(out_ch, in_ch * 2 + out_ch)

        # Init: trunc_normal(0.02); adaLN zero (adaLN-zero gating)
        for m in (self.dwconv, self.pwconv1, self.pwconv2):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        if isinstance(self.shortcut, nn.Conv2d):
            nn.init.trunc_normal_(self.shortcut.weight, std=0.02)
            nn.init.zeros_(self.shortcut.bias)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, t_emb):
        shortcut = self.shortcut(x)
        h = self.norm(self.dwconv(x))
        shift, scale, gate = self.adaLN(t_emb).split([self.in_ch, self.in_ch, self.out_ch], dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = h.permute(0, 2, 3, 1)
        h = self.pwconv2(self.act(self.pwconv1(h)))
        h = h.permute(0, 3, 1, 2)
        return shortcut + gate[:, :, None, None] * h


class UpsampleBlock4x(nn.Module):
    """Learned 4x upsample (2 x conv + pixel shuffle). target_hw trims the output to y_hat's unpadded size: z_hat
    is z-padded, so the upsample is up to 3 px larger than y_hat for sizes not divisible by 4."""

    def __init__(self, ch):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
            nn.Conv2d(ch, ch * 4, 3, padding=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, target_hw=None):
        out = self.layers(x)
        out = out[:, :, :target_hw[0], :target_hw[1]] if target_hw is not None else out
        return out


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of NCHW tensors (ConvNeXt channels_first), without affine parameters."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), (self.dim,),
                            None, None, self.eps).permute(0, 3, 1, 2)


def _make_head(hidden_ch, latent_ch):
    head = nn.Sequential(
        LayerNorm2d(hidden_ch),
        nn.Conv2d(hidden_ch, latent_ch, 1)
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def _to_symbol(x, q_dec):
    return x / q_dec.clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class PerQCompensatorBase(nn.Module):
    """_predict() returns an unclamped delta; sample() applies the bound.
    bound = 0.5 * bound_scale in symbol space, applied as bound * tanh(delta / bound).
    bound_scale < 1 restricts the delta to a sub-region of the quantization cell."""

    def __init__(self, clamp_unit_bound, bound_mode, bound_scale):
        super().__init__()
        assert clamp_unit_bound is True, f'clamp_unit_bound must be true, got {clamp_unit_bound!r}'
        assert bound_mode == 'tanh', f'bound_mode must be "tanh", got {bound_mode!r}'
        assert 0.0 < float(bound_scale) <= 1.0, \
            f'bound_scale must be in (0, 1], got {bound_scale!r}'
        self.bound_scale = float(bound_scale)

    def _build_trunk(self, cond_ch, hidden_ch, num_blocks, expansion):
        """Registers z_up, t_embed and blocks, in this order (it fixes the init RNG order)."""
        self.z_up = UpsampleBlock4x(cond_ch)
        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_ch),
            nn.Linear(hidden_ch, hidden_ch),
            nn.SiLU()
        )
        nn.init.trunc_normal_(self.t_embed[1].weight, std=0.02)
        nn.init.zeros_(self.t_embed[1].bias)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(hidden_ch, hidden_ch, expansion=expansion) for _ in range(num_blocks)]
        )

    def _run_blocks(self, h, t_emb):
        for block in self.blocks:
            h = block(h, t_emb)
        return h

    def _apply_bound(self, delta):
        bound = self.bound_scale * 0.5
        return bound * torch.tanh(delta / bound)

    def _predict(self, y_hat, z_hat, q_dec, num_steps=1, noise=None):
        raise NotImplementedError

    def sample(self, y_hat, z_hat, q_dec, num_steps=1, noise=None):
        delta = self._predict(y_hat, z_hat, q_dec, num_steps=num_steps, noise=noise)
        return self._apply_bound(delta)

    def compute_loss(self, y_hat, z_hat, q_dec, delta_target):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Flow Matching Compensator
# ---------------------------------------------------------------------------

class FlowMatchingCompensator(PerQCompensatorBase):
    """Conditional flow matching (Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023) with the
    rectified-flow straight path (Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with
    Rectified Flow", ICLR 2023). Trains velocity v_θ(x_t, t, cond) along straight-line paths from x_0 ~ N(0, σ²·I)
    to delta_target; inference draws x_0 ~ N(0, σ²·I) and integrates with N-step Euler. The network works in
    q_dec-scaled space (y_hat * q_dec in, delta / q_dec out); the loss is computed in symbol space."""

    def __init__(self, latent_ch=256, cond_ch=128, hidden_ch=512, num_blocks=10,
                 expansion=4, clamp_unit_bound=True, bound_mode='tanh',
                 bound_scale=0.1, noise_sigma=0.029, zero_init_head=True,
                 use_dec_scale=True, use_dec_scale_loss=False, block_type='convnext',
                 z_upsample='learned'):
        super().__init__(clamp_unit_bound=clamp_unit_bound, bound_mode=bound_mode, bound_scale=bound_scale)
        assert zero_init_head is True, f'zero_init_head must be true, got {zero_init_head!r}'
        assert use_dec_scale is True, f'use_dec_scale must be true, got {use_dec_scale!r}'
        assert use_dec_scale_loss is False, f'use_dec_scale_loss must be false, got {use_dec_scale_loss!r}'
        assert block_type == 'convnext', f'block_type must be "convnext", got {block_type!r}'
        assert z_upsample == 'learned', f'z_upsample must be "learned", got {z_upsample!r}'
        self.noise_sigma = float(noise_sigma)

        # The zero-init head and adaLN-zero gates make the output 0 at initialisation
        self.cond_proj = nn.Conv2d(latent_ch * 2 + cond_ch, hidden_ch, 1)
        nn.init.trunc_normal_(self.cond_proj.weight, std=0.02)
        nn.init.zeros_(self.cond_proj.bias)
        self.x_proj = nn.Conv2d(latent_ch, hidden_ch, 1)
        nn.init.trunc_normal_(self.x_proj.weight, std=0.02)
        nn.init.zeros_(self.x_proj.bias)
        self._build_trunk(cond_ch, hidden_ch, num_blocks, expansion)
        self.head = _make_head(hidden_ch, latent_ch)

    def _velocity(self, x_t, t, y_hat, z_hat_up, q_dec):
        h = self.cond_proj(torch.cat([y_hat, z_hat_up, q_dec], dim=1)) + self.x_proj(x_t)
        t_emb = self.t_embed(t)
        return self.head(self._run_blocks(h, t_emb))

    def _predict(self, y_hat, z_hat, q_dec, num_steps=1, noise=None):
        assert num_steps >= 1, f'FlowMatching num_steps must be >= 1, got {num_steps}'
        noise = self.noise_sigma * torch.randn_like(y_hat) if noise is None else noise
        z_hat_up = self.z_up(z_hat, target_hw=y_hat.shape[-2:])

        noise = q_dec * noise
        y_hat = q_dec * y_hat

        dt = 1.0 / num_steps
        x = noise
        for i in range(num_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            x = x + self._velocity(x, t, y_hat, z_hat_up, q_dec) * dt

        return _to_symbol(x, q_dec)

    def compute_loss(self, y_hat, z_hat, q_dec, delta_target):
        noise_sym = self.noise_sigma * torch.randn_like(delta_target)
        delta_target_sym = delta_target
        z_hat_up = self.z_up(z_hat, target_hw=y_hat.shape[-2:])

        noise, y_hat, delta_target = q_dec * noise_sym, q_dec * y_hat, q_dec * delta_target

        t = torch.rand(y_hat.shape[0], device=y_hat.device, dtype=y_hat.dtype)
        x_t = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * delta_target
        v_pred = self._velocity(x_t, t, y_hat, z_hat_up, q_dec)
        # Symbol-space loss with a q_dec-scaled network: divide the prediction by q_dec before the MSE
        return F.mse_loss(_to_symbol(v_pred, q_dec), delta_target_sym - noise_sym)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_perq_compensator(perq_type, **kwargs):
    if perq_type == 'flow_matching':
        return FlowMatchingCompensator(**kwargs)
    else:
        raise ValueError(f'perq_type must be "flow_matching", got {perq_type!r}')