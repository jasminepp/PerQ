"""
IO utilities
"""
from perq.utils import *


def compute_scale_shift(channels, bit_depth, scale, shift):
    scale = torch.tensor(scale if scale is not None else [1.] * channels, dtype=torch.float32) / (2 ** bit_depth - 1)
    shift = torch.tensor(shift if shift is not None else [0.] * channels, dtype=torch.float32)
    return scale, shift