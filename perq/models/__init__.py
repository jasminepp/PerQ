from . import backbone
from . import perq


def create_compress_model(logger, model_cfg, num_rates):
    if model_cfg.model_type == 'dcvc-rt_custom_intra':
        return backbone.DCVC_RT_Custom_Intra(logger, model_cfg.model_cfg, num_rates)
    elif model_cfg.model_type == 'dcvc-rt_perq_intra':
        return perq.DCVC_RT_PerQ_Intra(logger, model_cfg.model_cfg, num_rates)
    else:
        raise ValueError(f'Unknown compress model type: {model_cfg.model_type}')