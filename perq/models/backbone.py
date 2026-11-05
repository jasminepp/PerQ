"""
DCVC-RT intra codec (Jia et al., "Towards Practical Real-Time Neural Video Compression", CVPR 2025) with bottleneck
quantisation scales: the PerQ backbone
"""
# Adapted from DCVC-RT (https://github.com/microsoft/DCVC), Copyright (c) Microsoft Corporation, MIT License.
from perq.utils import *
from .codec_base import *
from perq.entropy.quantizer import get_quantizer
from perq.entropy.distributions import GaussianConditional
from perq.entropy.utils import clip_with_grad

from third_party.DCVC.src.models.common_model import CompressionModel
from third_party.DCVC.src.models.image_model import g_ch_src
from third_party.DCVC.src.layers.layers import SubpelConv2x, DepthConvBlock, ResidualBlockWithStride2, ResidualBlockUpsample
from third_party.DCVC.src.layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE
from perq.gan.gan_module import GANModule, GANConfig


def _check_infer_dtype(name):
    assert name in ('fp32', 'fp16'), \
        f'inference_dtype must be "fp32" | "fp16", got {name!r}'
    return name


def _snapshot_fp32_and_half(module):
    with torch.inference_mode(False), torch.no_grad():
        master = {k: v.detach().clone() for k, v in module.state_dict().items()}
        module.half()
    return master


def _restore_fp32(module, master):
    with torch.inference_mode(False), torch.no_grad():
        module.float()
        module.load_state_dict(master, strict=False)


class _DCVCRTBlocks:
    Residual = DepthConvBlock
    Downsample = ResidualBlockWithStride2
    Upsample = ResidualBlockUpsample


_BLOCK_CLSES = {
    'dcvc-rt': _DCVCRTBlocks,
}


def _get_blocks(cls):
    return _BLOCK_CLSES[cls]


# Compiled-in availability, snapshotted at import: the CUSTOMIZED_CUDA_INFERENCE global of cuda_inference is one of
# the globals disable_custom_ops() writes, so re-deriving availability per instance would latch a later model to
# unavailable for its whole lifetime.
_CUSTOM_OPS_COMPILED = CUSTOMIZED_CUDA_INFERENCE


def _get_custom_ops_modules():
    _modules = []
    for _mod_name in ['third_party.DCVC.src.layers.cuda_inference',
                      'third_party.DCVC.src.layers.layers',
                      'third_party.DCVC.src.models.image_model']:
        _mod = sys.modules.get(_mod_name)
        if _mod is not None and hasattr(_mod, 'CUSTOMIZED_CUDA_INFERENCE'):
            _modules.append(_mod)
    return _modules


def _make_res_stack(first_in, channels, depth, blocks, tail=None):
    assert depth >= 1, f'res stack depth must be >= 1 (got {depth})'
    layers = [blocks.Residual(first_in, channels)] + \
             [blocks.Residual(channels, channels) for _ in range(depth - 1)]
    if tail is not None:
        layers.append(tail())
    return nn.Sequential(*layers)


def _make_hyper_enc_stack(in_channels, z_channels, depth, blocks):
    assert depth >= 2, f'hyper_enc depth must be >= 2 (got {depth})'
    assert depth >= 3 or in_channels == z_channels, \
        f'hyper_enc depth == 2 has no head Residual; the first Downsample takes the raw ' \
        f'{in_channels}-channel input but expects z_channels={z_channels}'
    return nn.Sequential(*([blocks.Residual(in_channels if i == 0 else z_channels, z_channels)
                            for i in range(depth - 2)] + \
                           [blocks.Downsample(z_channels, z_channels), blocks.Downsample(z_channels, z_channels)]))


def _make_hyper_dec_stack(z_channels, out_channels, depth, blocks):
    assert depth >= 2, f'hyper_dec depth must be >= 2 (got {depth})'
    assert depth >= 3 or out_channels == z_channels, \
        f'hyper_dec depth == 2 has no tail Residual to switch channels; output stays ' \
        f'{z_channels}-channel but out_channels={out_channels}'
    return nn.Sequential(*([blocks.Upsample(z_channels, z_channels), blocks.Upsample(z_channels, z_channels)] + \
                           [blocks.Residual(z_channels, out_channels if i == depth - 3 else z_channels)
                            for i in range(depth - 2)]))


def _make_dist_model():
    scale_table = np.exp(np.linspace(np.log(0.11), np.log(256), 64)).tolist()
    return GaussianConditional(scale_table=scale_table, scale_bound=0.1)


def _make_rp_to_qp(num_rates, qp_num):
    qp_list = [int(rp * (qp_num - 1) / max(num_rates - 1, 1)) for rp in range(num_rates)]
    return torch.tensor(qp_list, dtype=torch.long)


@dataclass
class DCVCRTCoreIntraConfig:
    y_channels: int = 256
    z_channels: int = 128
    enc_channels: int = 368
    dec_channels: int = 368
    enc_depth: int = 6
    dec_depth: int = 12
    hyper_enc_depth: int = 3
    hyper_dec_depth: int = 3
    prior_depth: int = 3
    enc_block: str = 'dcvc-rt'
    dec_block: str = 'dcvc-rt'
    hyper_enc_block: str = 'dcvc-rt'
    hyper_dec_block: str = 'dcvc-rt'
    qp_num: int = 64
    q_scale_mode: str = 'bottleneck-pre-prior'
    q_scale_pair: bool = True
    q_scale_mono: bool = True
    q_scale_bound: list = field(default_factory=lambda: [0.025, 40.0])
    use_log_scale: bool = True
    rate_quant: str = 'uniform-noise'
    dist_quant: str = 'ste'
    em_dist: str = 'gaussian'

    def __post_init__(self):
        for name in ('enc_block', 'dec_block', 'hyper_enc_block', 'hyper_dec_block'):
            assert getattr(self, name) == 'dcvc-rt', f'{name} must be "dcvc-rt", got {getattr(self, name)!r}'
        assert self.q_scale_mode == 'bottleneck-pre-prior', \
            f'q_scale_mode must be "bottleneck-pre-prior", got {self.q_scale_mode!r}'
        assert self.q_scale_pair is True, f'q_scale_pair must be true, got {self.q_scale_pair!r}'
        assert self.q_scale_mono is True, f'q_scale_mono must be true, got {self.q_scale_mono!r}'
        assert isinstance(self.q_scale_bound, (list, tuple)) and len(self.q_scale_bound) == 2, \
            f'q_scale_bound must be [lo, hi], got {self.q_scale_bound!r}'
        assert 0.0 < self.q_scale_bound[0] < self.q_scale_bound[1], \
            f'q_scale_bound must satisfy 0 < lo < hi, got {self.q_scale_bound!r}'
        assert self.use_log_scale is True, f'use_log_scale must be true, got {self.use_log_scale!r}'
        assert self.rate_quant == 'uniform-noise', f'rate_quant must be "uniform-noise", got {self.rate_quant!r}'
        assert self.dist_quant == 'ste', f'dist_quant must be "ste", got {self.dist_quant!r}'
        assert self.em_dist == 'gaussian', f'em_dist must be "gaussian", got {self.em_dist!r}'


@dataclass
class DCVCCustomIntraConfig:
    _MODEL_CFG_CLS = DCVCRTCoreIntraConfig
    model_cfg: dict = None
    gan_cfg: dict = None
    init_type: str = 'default'
    eval_with_compression: bool = True
    load_pretrain_gan: bool = False
    inference_dtype: str = 'fp16'
    fmt: str = 'rgb444'

    def __post_init__(self):
        if self.model_cfg is None:
            self.model_cfg = self._MODEL_CFG_CLS()
        elif isinstance(self.model_cfg, dict):
            self.model_cfg = self._MODEL_CFG_CLS(**self.model_cfg)
        assert self.init_type == 'default', f'init_type must be "default", got {self.init_type!r}'
        assert self.load_pretrain_gan is False, f'load_pretrain_gan must be false, got {self.load_pretrain_gan!r}'
        assert self.fmt == 'rgb444', f'fmt must be "rgb444", got {self.fmt!r}'


class _ConditionModel(CompressionModel):
    """
    Latent codec: rp_to_qp, quantizers and distribution model, the quantization / rate math and the 4-step
    checkerboard prior loop. Subclasses set self.cfg before calling super().__init__.
    """
    PAD_STRIDE = 16

    def __init__(self, num_rates):
        super().__init__(z_channel=self.cfg.z_channels, extra_qp=self._compute_extra_qp())

        # Rate-point to QP mapping
        self.register_buffer('rp_to_qp', _make_rp_to_qp(num_rates, self.cfg.qp_num))

        # Quantizers
        self.rate_quantizer = get_quantizer(self.cfg.rate_quant)
        self.dist_quantizer = get_quantizer(self.cfg.dist_quant)

        # Distribution model
        self.dist_model = _make_dist_model()

        # Built in this order: it fixes the initialisation RNG stream
        self.enc = self._make_encoder()
        self.dec = self._make_decoder()
        self.hyper_enc = self._make_hyper_encoder()
        self.hyper_dec = self._make_hyper_decoder()
        self.y_prior_fusion = self._make_prior_fusion()
        self._make_spatial_prior()
        self._make_q_params()

    def _compute_extra_qp(self):
        raise NotImplementedError

    def _make_encoder(self):
        raise NotImplementedError

    def _make_decoder(self):
        raise NotImplementedError

    def _make_hyper_encoder(self):
        raise NotImplementedError

    def _make_hyper_decoder(self):
        raise NotImplementedError

    def _make_prior_fusion(self):
        raise NotImplementedError

    def _make_spatial_prior(self):
        raise NotImplementedError

    def _make_q_params(self):
        raise NotImplementedError

    def _rate_math_ctx(self, device_type):
        # Rate math always runs in fp32: the 1e-9 probability floor underflows in
        # fp16 (-> log2(0) = -inf) and near-saturated CDF differences cancel.
        return torch.amp.autocast(device_type, enabled=False)

    def _q_init(self):
        p = max(min((1.0 - self.cfg.q_scale_bound[0]) /
                    (self.cfg.q_scale_bound[1] - self.cfg.q_scale_bound[0]), 1.0 - 1e-6), 1e-6)
        return float(math.log(p / (1.0 - p)))

    def _bound_q(self, q):
        return torch.sigmoid(q) * (self.cfg.q_scale_bound[1] - self.cfg.q_scale_bound[0]) + self.cfg.q_scale_bound[0]

    def _make_mono(self, q_raw, increasing=True):
        base = q_raw[:1]
        deltas = F.softplus(q_raw[1:])
        q_inc = torch.cat([base, base + torch.cumsum(deltas, dim=0)], dim=0)
        return q_inc if increasing else q_inc.flip(dims=[0])

    def _transform_scales(self, scales):
        return torch.exp(clip_with_grad(scales, -6., 6.))

    def _quantize_z(self, z, qp, compute_rates):
        # Rates are per-sample [N]; the wrapper reduces them.
        dtype = z.dtype
        with self._rate_math_ctx(z.device.type):
            z = z.float()
            z_hat = self.dist_quantizer(z, {}).clamp(-128, 127)
            rate_z = torch.zeros(z.shape[0], device=z.device)
            if compute_rates:
                z_rate = self.rate_quantizer(z, {})
                cdf_hi = self.bit_estimator_z.get_cdf(z_rate + 0.5, qp.int())
                cdf_lo = self.bit_estimator_z.get_cdf(z_rate - 0.5, qp.int())
                rate_z = -torch.log2((cdf_hi - cdf_lo).clamp(min=1e-9)).sum(dim=(1, 2, 3))
        return z_hat.to(dtype), rate_z

    def _diff_process_with_mask(self, y_scaled, scales, means, mask, compute_rates):
        dtype = y_scaled.dtype
        with self._rate_math_ctx(y_scaled.device.type):
            y_scaled, scales, means = y_scaled.float(), scales.float(), means.float()
            scales_hat = self._transform_scales(scales) * mask
            means_hat = means * mask
            y_res = (y_scaled - means_hat) * mask

            y_q_dist = self.dist_quantizer(y_res, {}).clamp(-128, 127)
            y_hat = y_q_dist + means_hat

            if compute_rates:
                y_q_rate = self.rate_quantizer(y_res, {})
                p = self.dist_model(y_q_rate, means=None, scales=scales_hat)
                rate = (-torch.log2(p.clamp(min=1e-9)) * mask).sum(dim=(1, 2, 3))
            else:
                rate = torch.zeros(y_scaled.shape[0], device=y_scaled.device)

        return y_hat.to(dtype), rate

    def _spatial_prior_adaptors(self):
        raise NotImplementedError

    def _forward_prior(self, y_scaled, scales, means, common, compute_rates):
        # Checkerboard prior: step 0 uses the fused-prior scales/means, each
        # later step re-predicts them from the decoded-so-far latent + common context.
        B, C, H, W = y_scaled.shape
        masks = self.get_mask_4x(B, C, H, W, y_scaled.dtype, y_scaled.device)
        adaptors = self._spatial_prior_adaptors()
        rate_y = torch.zeros(B, device=y_scaled.device)
        y_hat_so_far = None
        for i, mask in enumerate(masks):
            if i > 0:
                p = torch.cat((y_hat_so_far.to(common.dtype), common), dim=1)
                p = adaptors[i - 1](p)
                scales, means = self.y_spatial_prior(p).chunk(2, 1)
            y_hat_i, rate_i = self._diff_process_with_mask(y_scaled, scales, means, mask, compute_rates)
            rate_y = rate_y + rate_i
            y_hat_so_far = y_hat_i if i == 0 else y_hat_so_far + y_hat_i
        return y_hat_so_far, rate_y

    def _get_q_bott(self, qp):
        q_param = self._make_mono(self.q_scale, increasing=True)
        q_enc = self._bound_q(q_param[qp])
        q_dec = 1.0 / (q_enc + 1.0e-6)
        return q_enc, q_dec

    def encode_latent(self, x):
        return self.enc(x)

    def decode_latent(self, y_hat, clamp=True):
        out = self.dec(y_hat)
        return out.clamp_(0, 1) if clamp else out

    def _fuse_prior(self, z_hat, y_shape):
        # Fuse, then crop to the y dims; the crop comes after the fusion (conv boundary context).
        params = self.y_prior_fusion(self.hyper_dec(z_hat))
        return params[:, :, :y_shape[0], :y_shape[1]].contiguous()

    def _split_prior(self, params):
        raise NotImplementedError

    def forward_latent(self, y, qp, compute_rates=True):
        q_enc_bott, q_dec_bott = self._get_q_bott(qp)

        # Hyper encode (with the pre-prior bottleneck scaling on y).
        y_pad = self.pad_for_y(y * q_enc_bott)
        z = self.hyper_enc(y_pad)
        z_hat, rate_z = self._quantize_z(z, qp, compute_rates)

        params = self._fuse_prior(z_hat, y.shape[-2:])
        q_enc_local, q_dec_local, init_scales, init_means, common = self._split_prior(params)

        q_enc = q_enc_local * q_enc_bott
        q_dec = q_dec_local * q_dec_bott

        y_scaled = y * q_enc
        y_hat, rate_y = self._forward_prior(y_scaled, init_scales, init_means, common, compute_rates)

        stats = {'q_dec': q_dec, 'z_hat': z_hat,
                 'y_scaled': y_scaled, 'y_hat': y_hat,
                 'rates': {'y': rate_y, 'z': rate_z}}
        return stats

    def _restore_y(self, y_q, means, mask):
        from third_party.DCVC.src.layers.cuda_inference import restore_y_4x
        return restore_y_4x(y_q, means, mask)

    def _compress_prior(self, y, scales, means, common, q_enc):
        B, C, H, W = y.shape
        masks = self.get_mask_4x(B, C, H, W, y.dtype, y.device)
        adaptors = self._spatial_prior_adaptors()
        y = y * q_enc
        y_q_ws, s_ws, y_hat_so_far = [], [], None
        for i, mask in enumerate(masks):
            if i > 0:
                p = torch.cat((y_hat_so_far.to(common.dtype), common), dim=1)
                p = adaptors[i - 1](p)
                scales, means = self.y_spatial_prior(p).chunk(2, 1)
            _, y_q_i, y_hat_i, s_hat_i = self.process_with_mask(y, self._transform_scales(scales), means, mask)
            # Fresh compacted copies per step — the coder's index build clamps in place.
            y_q_ws.append(self.single_part_for_writing_4x(y_q_i))
            s_ws.append(self.single_part_for_writing_4x(s_hat_i))
            y_hat_so_far = y_hat_i if i == 0 else y_hat_so_far + y_hat_i
        return y_q_ws, s_ws, y_hat_so_far

    def _decompress_prior(self, scales, means, common):
        dtype, device = means.dtype, means.device
        B, C, H, W = means.shape
        masks = self.get_mask_4x(B, C, H, W, dtype, device)
        adaptors = self._spatial_prior_adaptors()
        y_hat_so_far = None
        for i, mask in enumerate(masks):
            if i > 0:
                p = torch.cat((y_hat_so_far.to(common.dtype), common), dim=1)
                p = adaptors[i - 1](p)
                scales, means = self.y_spatial_prior(p).chunk(2, 1)
            scales_r = self.single_part_for_writing_4x(self._transform_scales(scales) * mask)
            y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
            y_hat_i = self._restore_y(y_q_r, means, mask)
            y_hat_so_far = y_hat_i if i == 0 else y_hat_so_far + y_hat_i
        return y_hat_so_far

    def compress_latent(self, y, qp):
        from third_party.DCVC.src.layers.cuda_inference import round_and_to_int8
        device = y.device
        q_enc_bott, _ = self._get_q_bott(qp)
        y_pad = self.pad_for_y(y * q_enc_bott)
        z = self.hyper_enc(y_pad)
        z_hat, z_hat_write = round_and_to_int8(z)

        params = self._fuse_prior(z_hat, y.shape[-2:])
        q_enc_local, _, scales, means, common = self._split_prior(params)
        # The channel-chunk views from separate_prior sit at a 2*H*W-element offset —
        # 16B-misaligned when both y dims are odd — and the inference-extension kernels
        # use vectorized aligned loads (an upstream DCVC-RT issue; the views are contiguous, so
        # only clone() re-aligns them).
        scales, means = scales.clone(), means.clone()
        q_enc = q_enc_local * q_enc_bott

        y_q_ws, s_ws, _ = self._compress_prior(y, scales, means, common, q_enc)

        # Stream order: z, then the four y parts
        self.entropy_coder.reset()
        self.bit_estimator_z.encode_z(z_hat_write, int(qp.view(-1)[0]))
        for y_q_w, s_w in zip(y_q_ws, s_ws):
            self.gaussian_encoder.encode_y(y_q_w, s_w)
        self.entropy_coder.flush()
        bit_stream = self.entropy_coder.get_encoded_stream()
        torch.cuda.synchronize(device=device)
        return {'bit_stream': bit_stream}

    def decompress_latent(self, bit_stream, sps):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        qp = sps['qp']
        _, q_dec_bott = self._get_q_bott(qp)

        self.entropy_coder.set_use_two_entropy_coders(sps['ec_part'] == 1)
        self.entropy_coder.set_stream(bit_stream)
        z_size = self.get_downsampled_shape(sps['height'], sps['width'], 64)
        y_height, y_width = self.get_downsampled_shape(sps['height'], sps['width'], 16)
        self.bit_estimator_z.decode_z(z_size, int(qp.view(-1)[0]))
        z_hat = self.bit_estimator_z.get_z(z_size, device, dtype)

        params = self._fuse_prior(z_hat, (y_height, y_width))
        _, q_dec_local, scales, means, common = self._split_prior(params)
        # Same alignment clone as compress_latent (misaligned views + vectorized kernels).
        scales, means = scales.clone(), means.clone()
        q_dec = q_dec_local * q_dec_bott

        y_hat = self._decompress_prior(scales, means, common)
        return {'y_hat': y_hat, 'z_hat': z_hat, 'q_dec': q_dec}


class _IntraEncoder(nn.Module):
    def __init__(self, enc_channels, y_channels, enc_depth, blocks):
        super().__init__()
        self.enc_1 = blocks.Residual(g_ch_src, enc_channels)
        self.enc_2 = _make_res_stack(enc_channels, enc_channels, enc_depth, blocks,
                                     tail=lambda: nn.Conv2d(enc_channels, y_channels, 3, stride=2, padding=1))

    def forward(self, x):
        out = F.pixel_unshuffle(x, 8)
        out = self.enc_1(out)
        return self.enc_2(out)


class _IntraDecoder(nn.Module):
    def __init__(self, y_channels, dec_channels, dec_depth, blocks):
        super().__init__()
        self.dec_1 = nn.Sequential(*([blocks.Upsample(y_channels, dec_channels)] + \
                                     [blocks.Residual(dec_channels, dec_channels) for _ in range(dec_depth)]))
        self.dec_2 = blocks.Residual(dec_channels, g_ch_src)

    def forward(self, x):
        out = self.dec_1(x)
        out = self.dec_2(out)
        out = F.pixel_shuffle(out, 8)
        return out


class IntraModel(_ConditionModel):
    def __init__(self, cfg: DCVCRTCoreIntraConfig, num_rates=1):
        self.cfg = cfg
        extra_qp = self._compute_extra_qp()
        assert extra_qp == 0, f'extra_qp > 0 is not tested; got {extra_qp} from qp_num={self.cfg.qp_num}'
        super().__init__(num_rates=num_rates)

    def _compute_extra_qp(self):
        return self.cfg.qp_num - 64

    def _make_encoder(self):
        return _IntraEncoder(self.cfg.enc_channels, self.cfg.y_channels, self.cfg.enc_depth,
                             _get_blocks(self.cfg.enc_block))

    def _make_decoder(self):
        return _IntraDecoder(self.cfg.y_channels, self.cfg.dec_channels, self.cfg.dec_depth,
                             _get_blocks(self.cfg.dec_block))

    def _make_hyper_encoder(self):
        return _make_hyper_enc_stack(self.cfg.y_channels, self.cfg.z_channels,
                                     self.cfg.hyper_enc_depth, _get_blocks(self.cfg.hyper_enc_block))

    def _make_hyper_decoder(self):
        return _make_hyper_dec_stack(self.cfg.z_channels, self.cfg.y_channels,
                                     self.cfg.hyper_dec_depth, _get_blocks(self.cfg.hyper_dec_block))

    def _make_prior_fusion(self):
        return _make_res_stack(self.cfg.y_channels, self.cfg.y_channels * 2,
                               self.cfg.prior_depth, _get_blocks('dcvc-rt'),
                               tail=lambda: nn.Conv2d(self.cfg.y_channels * 2, self.cfg.y_channels * 2 + 2, 1))

    def _make_spatial_prior(self):
        # Reduction + 3 adaptors + shared spatial prior
        blocks = _get_blocks('dcvc-rt')
        self.y_spatial_prior_reduction = nn.Conv2d(self.cfg.y_channels * 2 + 2, self.cfg.y_channels, 1)
        self.y_spatial_prior_adaptor_1 = blocks.Residual(self.cfg.y_channels * 2, self.cfg.y_channels * 2,
                                                         force_adaptor=True)
        self.y_spatial_prior_adaptor_2 = blocks.Residual(self.cfg.y_channels * 2, self.cfg.y_channels * 2,
                                                         force_adaptor=True)
        self.y_spatial_prior_adaptor_3 = blocks.Residual(self.cfg.y_channels * 2, self.cfg.y_channels * 2,
                                                         force_adaptor=True)
        self.y_spatial_prior = _make_res_stack(self.cfg.y_channels * 2, self.cfg.y_channels * 2,
                                               self.cfg.prior_depth, blocks,
                                               tail=lambda: nn.Conv2d(self.cfg.y_channels * 2, self.cfg.y_channels * 2, 1))

    def _make_q_params(self):
        q_shape = (self.cfg.qp_num, self.cfg.y_channels, 1, 1)
        q_init = self._q_init()
        self.q_scale = nn.Parameter(torch.full(q_shape, q_init))

    def separate_prior(self, params, eps=1e-6):
        # Paired-q image layout: q from channel 0, scales/means from params[:, 2:] — odd-channel-offset views, the
        # 16B-misalignment hazard behind the clone-after-_split_prior in compress/decompress.
        q_raw = params[:, 0:1, :, :]
        q_enc = torch.sigmoid(q_raw) * 1.5 + 0.5
        q_dec = 1.0 / (q_enc + eps)
        scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means

    def _split_prior(self, params):
        q_enc, q_dec, scales, means = self.separate_prior(params)
        return q_enc, q_dec, scales, means, self.y_spatial_prior_reduction(params)

    def _spatial_prior_adaptors(self):
        return (self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3)


class DCVC_RT_Custom_Base(CodecBase):
    """
    DCVC-RT codec base: config, inference dtype, caches and pretrained-weight loading.
    """
    CONFIG_CLS = None

    def __init__(self, logger, cfg):
        super().__init__()
        assert self.CONFIG_CLS is not None, f'{type(self).__name__} must define CONFIG_CLS'
        self.logger = logger
        self.cfg = self.CONFIG_CLS(**cfg) if isinstance(cfg, dict) else cfg
        _check_infer_dtype(self.cfg.inference_dtype)
        self._init_custom_ops()

    def _init_custom_ops(self):
        # CUDA custom-op availability (disabled during training; enabled at eval)
        self._custom_ops_available = _CUSTOM_OPS_COMPILED
        self._custom_ops_enabled = True
        self._infer_halved = False
        self._fp32_master = None

    def get_fmt(self):
        return self.cfg.fmt

    @staticmethod
    def _unwrap_state_dict(state_dict):
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            if 'net' in state_dict:
                state_dict = state_dict['net']
            state_dict = copy.deepcopy(state_dict)
            nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')
        return state_dict

    def _infer_input_dtype(self):
        """The dtype eval inputs are cast to: the model's current param dtype. fp16 after
        _apply_infer_dtype has halved the net; fp32 otherwise (and always in training)."""
        return next(self.parameters()).dtype

    def _apply_infer_dtype(self):
        """Cast the net to fp16 for the eval forward (idempotent), for both the
        differentiable and the real-bitstream (compress/decompress) paths."""
        if self._infer_halved or self.cfg.inference_dtype != 'fp16':
            return
        self._fp32_master = _snapshot_fp32_and_half(self)
        self._clear_mask_caches()
        self._infer_halved = True

    def _restore_infer_dtype(self):
        if not self._infer_halved:
            return
        _restore_fp32(self, self._fp32_master)
        self._clear_mask_caches()
        self._fp32_master = None
        self._infer_halved = False

    def _clear_mask_caches(self):
        # DCVC-RT's mask cache (get_mask_4x) keys on shape, not dtype: clear it on every dtype change
        for m in self.modules():
            masks = getattr(m, 'masks', None)
            if isinstance(masks, dict):
                masks.clear()

    def _reset_inference_proxies(self):
        for m in self.modules():
            if isinstance(m, (DepthConvBlock, SubpelConv2x)):
                m.proxy = None

    def load_cache(self):
        self._apply_infer_dtype()
        self._reset_inference_proxies()

    def clear_cache(self):
        self._restore_infer_dtype()
        self._reset_inference_proxies()

    def enable_custom_ops(self):
        self._custom_ops_enabled = True
        for mod in _get_custom_ops_modules():
            mod.CUSTOMIZED_CUDA_INFERENCE = self._custom_ops_available

    def disable_custom_ops(self):
        self._custom_ops_enabled = False
        for mod in _get_custom_ops_modules():
            mod.CUSTOMIZED_CUDA_INFERENCE = False

    def _enable_custom_ops_if_enabled(self):
        if self._custom_ops_enabled:
            for mod in _get_custom_ops_modules():
                mod.CUSTOMIZED_CUDA_INFERENCE = self._custom_ops_available

    def _disable_custom_ops_if_enabled(self):
        if self._custom_ops_enabled:
            for mod in _get_custom_ops_modules():
                mod.CUSTOMIZED_CUDA_INFERENCE = False


class DCVC_RT_Custom_Intra(DCVC_RT_Custom_Base):
    """
    DCVC-RT intra codec
    """
    CONFIG_CLS = DCVCCustomIntraConfig

    def __init__(self, logger, cfg, num_rates):
        super().__init__(logger, cfg)

        # Build the actual codec
        self.intra_model = IntraModel(self.cfg.model_cfg, num_rates=num_rates)

        # GAN module
        if self.cfg.gan_cfg is not None:
            self.gan = GANModule(logger, GANConfig(**self.cfg.gan_cfg))
            logger.info(f'GAN module:')
            logger.info(f'  Discriminator: {self.gan.cfg.disc_type}')
            logger.info(f'  Num params: {sum(p.numel() for p in self.gan.parameters())/1e6:.2f}M')
        else:
            self.gan = None

        # Initialize the codec
        self.intra_model.update()

        num_params = sum(p.numel() for p in self.intra_model.parameters())
        logger.info(f'DCVC-RT intra component: {num_params/1e6:.2f}M params')

    def _latent_postprocess(self, states, compute_outputs):
        return states

    def _compute_aux_losses(self, states, x_hat, compute_outputs, compute_rates):
        model_losses = {}

        # Gan loss
        if self.gan is not None and compute_outputs and x_hat is not None:
            model_losses.update(self.gan.compute_losses(states['x_in'], x_hat, rp_idx=states['rp']))

        return model_losses

    def _forward(self, x_dict, compute_outputs=True, compute_rates=True, reduce_rates=True):
        assert x_dict['x'].shape[2] == 1, f'{type(self).__name__} codes one frame at a time'
        # Match the net's current dtype
        x_in = x_dict['x'].squeeze(2).to(self._infer_input_dtype())
        _, _, H, W = x_in.shape
        padding, unpadding = compressai.ops.compute_padding(H, W, min_div=self.intra_model.PAD_STRIDE)
        x_padded = F.pad(x_in, padding, mode='replicate')
        qp = self.intra_model.rp_to_qp[x_dict['rp'].view(-1).long()]

        # Encoding
        y = self.intra_model.encode_latent(x_padded)

        # Compress/decompress latent
        decompress_stats = self.intra_model.forward_latent(y, qp, compute_rates=compute_rates)
        decompress_stats.update({'x_in': x_in, 'unpadding': unpadding, 'rp': x_dict['rp']})

        # Postprocess latent
        latent_stats = self._latent_postprocess(decompress_stats, compute_outputs)

        x_hat = None
        if compute_outputs:
            y_hat = latent_stats['y_hat'] * latent_stats['q_dec']
            x_hat_4d = self.intra_model.decode_latent(y_hat, clamp=False)
            x_hat_4d = F.pad(x_hat_4d, latent_stats['unpadding'])
            x_hat = x_hat_4d.unsqueeze(2)

        # Rates
        rate = sum(latent_stats['rates'].values()).view(-1, 1) if reduce_rates else latent_stats['rates']
        rate_loss = rate

        # Model losses (GAN, etc).
        model_losses = self._compute_aux_losses(latent_stats, x_hat, compute_outputs, compute_rates)

        if compute_outputs and compute_rates:
            return x_hat, rate, rate_loss, model_losses
        elif compute_outputs:
            return x_hat, model_losses
        elif compute_rates:
            return rate, rate_loss, model_losses

    @torch.compiler.disable
    def _forward_via_compression(self, x_dict, compute_outputs=True, compute_rates=True, reduce_rates=True):
        # Parse + pad input.
        assert x_dict['x'].shape[2] == 1, f'{type(self).__name__} only supports T=1 (image)'
        x_in = x_dict['x'].squeeze(2).to(self._infer_input_dtype())
        _, _, H, W = x_in.shape
        padding, unpadding = compressai.ops.compute_padding(H, W, min_div=self.intra_model.PAD_STRIDE)
        x_padded = F.pad(x_in, padding, mode='replicate')

        # Encoding + compress to bit_stream
        qp = self.intra_model.rp_to_qp[x_dict['rp'].view(-1).long()]
        y = self.intra_model.encode_latent(x_padded)
        compress_stats = self.intra_model.compress_latent(y, qp)

        # Decompress to symbol-space latent (no decode yet)
        sps = {'height': x_padded.shape[-2], 'width': x_padded.shape[-1], 'ec_part': 0, 'qp': qp}
        decompress_stats = self.intra_model.decompress_latent(compress_stats['bit_stream'], sps)
        decompress_stats.update({'x_in': x_in, 'unpadding': unpadding, 'rp': x_dict['rp']})

        # Postprocess latent
        latent_stats = self._latent_postprocess(decompress_stats, compute_outputs)

        x_hat = None
        if compute_outputs:
            y_hat = latent_stats['y_hat'] * latent_stats['q_dec']
            x_hat_4d = self.intra_model.decode_latent(y_hat, clamp=False)
            x_hat_4d = F.pad(x_hat_4d, latent_stats['unpadding'])
            x_hat = x_hat_4d.unsqueeze(2)

        # Rates
        bits = torch.full((x_in.shape[0], 1), len(compress_stats['bit_stream']) * 8.0, device=x_in.device)
        rate = bits if reduce_rates else {'bits': bits}
        rate_loss = rate

        # Model losses (GAN, etc).
        model_losses = self._compute_aux_losses(latent_stats, x_hat, compute_outputs, compute_rates)

        if compute_outputs and compute_rates:
            return x_hat, rate, rate_loss, model_losses
        elif compute_outputs:
            return x_hat, model_losses
        elif compute_rates:
            return rate, rate_loss, model_losses

    def forward(self, x_dict, compute_outputs=True, compute_rates=True, reduce_rates=True):
        if self.training:
            # Training- custom ops disabled
            self._disable_custom_ops_if_enabled()
            return self._forward(x_dict, compute_outputs, compute_rates, reduce_rates)
        else:
            if x_dict['x'].shape[0] == 1:
                self._enable_custom_ops_if_enabled()
            else:
                # The inference-extension kernels assert batch == 1
                self._disable_custom_ops_if_enabled()
            with torch.amp.autocast('cuda', enabled=False):
                if self.cfg.eval_with_compression:
                    return self._forward_via_compression(x_dict, compute_outputs, compute_rates, reduce_rates)
                else:
                    return self._forward(x_dict, compute_outputs, compute_rates, reduce_rates)

    def load_cache(self):
        super().load_cache()
        self.intra_model.update()

    def compress(self, x, rp):
        with torch.amp.autocast('cuda', enabled=False):
            qp = self.intra_model.rp_to_qp[rp.view(-1).long()]
            # Encode latent
            y = self.intra_model.encode_latent(x.to(self._infer_input_dtype()))
            # Compress latent
            return self.intra_model.compress_latent(y, qp)

    def decompress(self, bit_stream, sps):
        with torch.amp.autocast('cuda', enabled=False):
            qp = self.intra_model.rp_to_qp[sps['rp'].view(-1).long()]
            # Decompress latent
            decompress_stats = self.intra_model.decompress_latent(bit_stream, {**sps, 'qp': qp})

            # Postprocess latents
            latent_stats = {'y_hat': decompress_stats['y_hat'],
                            'z_hat': decompress_stats['z_hat'],
                            'q_dec': decompress_stats['q_dec'],
                            'rp': sps['rp']}
            latent_stats = self._latent_postprocess(latent_stats, compute_outputs=True)
            y_hat = latent_stats['y_hat'] * latent_stats['q_dec']

            # decode latent
            return {'x_hat': self.intra_model.decode_latent(y_hat)}

    def set_stage(self, stage, num_stages):
        if self.gan is not None:
            self.gan.set_stage(stage, num_stages)

    def set_iter(self, i, n):
        if self.gan is not None:
            self.gan.set_iter(i, n)

    def load_pretrain(self, state_dict):
        # Checkpoints of this code (keys under intra_model.* [+ gan.*], possibly nested under
        # 'state_dict'/'net' and/or DDP-wrapped with 'module.').
        state_dict = self._unwrap_state_dict(state_dict)
        # rp_to_qp is a config-derived lookup (length=num_rates), not a learned param.
        # Drop it so the freshly-constructed buffer from current config wins — required
        # when fine-tuning at a different num_rp than the pretrain.
        state_dict.pop('intra_model.rp_to_qp', None)
        if not self.cfg.load_pretrain_gan:
            dropped = [k for k in list(state_dict.keys()) if k.startswith('gan.')]
            for k in dropped:
                state_dict.pop(k, None)
            if dropped:
                self.logger.info(f'load_pretrain: dropping {len(dropped)} `gan.*` keys '
                                 f'(load_pretrain_gan=False)')
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        return list(missing), list(unexpected)