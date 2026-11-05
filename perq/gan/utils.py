from perq.utils import *


# Hinge loss (Lim and Ye, "Geometric GAN", arXiv 2017), averaged over the score map and the masked samples
def hinge_d_loss(real_scores, fake_scores, mask):
    mask_b = mask.view(-1, *([1] * (real_scores.ndim - 1)))
    denom = mask.sum().clamp(min=1.0) * real_scores[0].numel()
    loss_real = (F.relu(1.0 - real_scores) * mask_b).sum() / denom
    loss_fake = (F.relu(1.0 + fake_scores) * mask_b).sum() / denom
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(fake_scores, mask):
    mask_b = mask.view(-1, *([1] * (fake_scores.ndim - 1)))
    denom = mask.sum().clamp(min=1.0) * fake_scores[0].numel()
    return -(fake_scores * mask_b).sum() / denom


def _get_g_loss_fn(name):
    _G_LOSS_FNS = {
        'hinge': hinge_g_loss,
    }
    return _G_LOSS_FNS[name]


def _get_d_loss_fn(name):
    _D_LOSS_FNS = {
        'hinge': hinge_d_loss,
    }
    return _D_LOSS_FNS[name]