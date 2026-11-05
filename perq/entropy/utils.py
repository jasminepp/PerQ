from perq.utils import *


class CustomClip(torch.autograd.Function):
    """
    See https://interdigitalinc.github.io/CompressAI/_modules/compressai/ops/bound_ops.html#LowerBound
    """
    @staticmethod
    def forward(ctx, x, lower, upper):
        ctx.save_for_backward(x)
        ctx.lower = lower
        ctx.upper = upper
        return torch.clamp(x, lower, upper)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        lower = ctx.lower
        upper = ctx.upper
        mask = ((x < lower) * (grad_output < 0)) + ((x > upper) * (grad_output > 0)) + (x >= lower) * (x <= upper)
        return mask * grad_output, None, None


def clip_with_grad(x, lower, upper):
    return CustomClip.apply(x, lower, upper)