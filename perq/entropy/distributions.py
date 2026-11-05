from .utils import *


class InnerGaussianConditional(compressai.entropy_models.GaussianConditional):
    pass


class GaussianConditional(nn.Module):
    """
    See https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/entropy_models/entropy_models.py
    """
    def __init__(self, scale_table=None, *args, **kwargs):
        nn.Module.__init__(self)
        self.inner_module = InnerGaussianConditional(scale_table, *args, **kwargs)
        self.inner_module.update()

    def forward(self, x, means, scales):
        p = self.inner_module._likelihood(x, means=means, scales=scales)
        if self.inner_module.use_likelihood_bound:
            p = self.inner_module.likelihood_lower_bound(p)
        return p