from .utils import *

try:
    import triton
    _TRITON_VERSION = tuple(int(v) for v in triton.__version__.split('.')[:2])
except Exception:
    _TRITON_VERSION = None


# triton < 3.8 can fail to compile kernels that fuse tl.rand (triton-lang/triton#8257): draw the noise outside them
_TRITON_RAND_FUSION_BUG = _TRITON_VERSION is not None and _TRITON_VERSION < (3, 8)


def _uniform_noise(inputs):
    return torch.rand_like(inputs) - 0.5


if _TRITON_RAND_FUSION_BUG:
    _uniform_noise = torch.compiler.disable(_uniform_noise)


class QuantizerBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, params):
        raise NotImplementedError


class UniformNoiseQuantizer(QuantizerBase):
    """Additive uniform noise as the training proxy of rounding (Ballé et al., "End-to-end Optimized Image
    Compression", ICLR 2017)."""
    def forward(self, inputs, params):
        if self.training:
            return inputs + _uniform_noise(inputs)
        else:
            return inputs.round()


class STEQuantizer(QuantizerBase):
    """Rounding with the straight-through gradient."""
    def forward(self, inputs, params):
        if self.training:
            return compressai.ops.quantize_ste(inputs)
        else:
            return inputs.round()


def get_quantizer(quantizer):
    if quantizer == 'uniform-noise':
        return UniformNoiseQuantizer()
    elif quantizer == 'ste':
        return STEQuantizer()
    else:
        raise ValueError