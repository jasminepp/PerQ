"""
DCVC-RT intra codec with a PerQ compensator
"""
from perq.utils import *
from .backbone import DCVCCustomIntraConfig, DCVC_RT_Custom_Intra
from .compensator import build_perq_compensator


@dataclass
class DCVCPerQConfig(DCVCCustomIntraConfig):
    # Compensator
    perq_type: str = 'flow_matching'
    perq_kwargs: dict = None
    # Euler steps in training (one entry)
    perq_train_steps: list = field(default_factory=lambda: [4])
    perq_inference_steps: int = 4
    # Latent (flow-matching) loss weight; 0 turns the loss off
    perq_latent_loss_scale: float = 1.0e4
    # Freezes every intra_model parameter; the compensator and GAN stay trainable
    freeze_backbone: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.perq_type == 'flow_matching', f'perq_type must be "flow_matching", got {self.perq_type!r}'
        assert isinstance(self.perq_train_steps, list) and len(self.perq_train_steps) == 1 and \
            all(n >= 1 for n in self.perq_train_steps) and self.perq_inference_steps >= 1, \
            f'perq_train_steps must be a one-entry list of a step count >= 1 and perq_inference_steps >= 1, ' \
            f'got {self.perq_train_steps!r} / {self.perq_inference_steps!r}'
        assert self.freeze_backbone is True, f'freeze_backbone must be true, got {self.freeze_backbone!r}'


class DCVC_RT_PerQ_Intra(DCVC_RT_Custom_Intra):
    """DCVC-RT intra codec with PerQ compensation"""
    CONFIG_CLS = DCVCPerQConfig

    def __init__(self, logger, cfg, num_rates):
        super().__init__(logger, cfg, num_rates)

        perq_kwargs = {'latent_ch': self.cfg.model_cfg.y_channels,
                       'cond_ch':   self.cfg.model_cfg.z_channels}
        perq_kwargs.update(self.cfg.perq_kwargs or {})
        self.perq_compensator = build_perq_compensator(self.cfg.perq_type, **perq_kwargs)

        # Frozen backbone
        for p in self.intra_model.parameters():
            p.requires_grad_(False)

        num_perq_params = sum(p.numel() for p in self.perq_compensator.parameters())
        num_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_train_backbone = sum(p.numel() for p in self.intra_model.parameters() if p.requires_grad)
        logger.info(f'PerQ compensator ({self.cfg.perq_type}): {num_perq_params/1e6:.2f}M params')
        logger.info(f'Freeze backbone: {self.cfg.freeze_backbone}')
        logger.info(f'GAN module present: {self.gan is not None}')
        logger.info(f'Trainable backbone: {num_train_backbone/1e6:.2f}M  '
                    f'Total trainable: {num_train_params/1e6:.2f}M')

    def _apply_infer_dtype(self):
        # The backbone runs in its own inference dtype, so it codes the same bitstream as the plain backbone; the
        # compensator stays in fp32
        super()._apply_infer_dtype()
        if self._infer_halved:
            with torch.inference_mode(False), torch.no_grad():
                self.perq_compensator.float()
                self.perq_compensator.load_state_dict({k[len('perq_compensator.'):]: v for k, v in
                                                       self._fp32_master.items() if k.startswith('perq_compensator.')})

    def _latent_postprocess(self, states, compute_outputs):
        states = super()._latent_postprocess(states, compute_outputs)

        if not compute_outputs and not self.training:
            return states

        with torch.amp.autocast(states['y_hat'].device.type, enabled=False):
            y_hat = states['y_hat'].float()
            z_hat = states['z_hat'].float()
            q_dec = states['q_dec'].float().expand(y_hat.shape[0], -1, *y_hat.shape[-2:])

            # Compensator sample
            if compute_outputs:
                num_steps = self.cfg.perq_train_steps[0] if self.training else self.cfg.perq_inference_steps
                delta = self.perq_compensator.sample(y_hat, z_hat, q_dec, num_steps=num_steps)
                states['y_hat'] = (y_hat + delta).to(states['y_hat'].dtype)

            # Latent loss
            if self.training and self.cfg.perq_latent_loss_scale > 0:
                perq_latent_loss_scale = self.cfg.perq_latent_loss_scale
                delta_target = states['y_scaled'].float() - y_hat
                states['perq_latent_loss'] = perq_latent_loss_scale * \
                                             self.perq_compensator.compute_loss(y_hat, z_hat, q_dec, delta_target)
            else:
                states['perq_latent_loss'] = None

        return states

    def _compute_aux_losses(self, states, x_hat, compute_outputs, compute_rates):
        model_losses = super()._compute_aux_losses(states, x_hat, compute_outputs, compute_rates)
        if states.get('perq_latent_loss') is not None:
            model_losses['perq_latent'] = states['perq_latent_loss']
        return model_losses