"""
Base class for compression models
"""
from perq.utils import *


class CodecBase(nn.Module):
    """
    Codec base class
    """
    coding_mode = 'chunk'

    def get_coding_mode(self):
        """
        Get coding mode
        """
        return self.coding_mode

    def clear_buffer(self):
        """
        Clear sequential coding state
        """
        pass

    def set_stage(self, stage, num_stages):
        """
        Called once at the start of each training stage.
        """
        pass

    def set_iter(self, iter, num_iters):
        """
        Set iter
        """
        raise NotImplementedError

    def load_pretrain(self, model):
        """
        Load pretrain
        """
        raise NotImplementedError