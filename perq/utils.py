import os
import sys

# For reproducibility
if "torch" in sys.modules and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
    raise RuntimeError(
        "torch should not be imported before setting CUBLAS_WORKSPACE_CONFIG for reproducibility."
    )
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import gc
import shutil
import argparse
import yaml
import json
import copy
import logging
import time
import datetime
import uuid
import fnmatch
from typing import Union, List, Optional, Any, Tuple
import dataclasses
from dataclasses import dataclass, field
from contextlib import nullcontext

import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo as dynamo

import torchvision
import compressai

from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

import accelerate


# Default parameter type, read by the weight-decay filter of the optimizer
nn.Parameter.param_type = 'default'


"""
Tools
"""
def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError


def load_yaml_config(config):
    if isinstance(config, str):
        if config is not None:
            with open(config, 'r') as f:
                return yaml.safe_load(f) or {}
        else:
            return {}
    else:
        return config or {}


def get_output_path(args):
    args.output_dir = os.path.expanduser(args.output_dir)
    if not args.exp_name:
        args.exp_name = '-'.join(
            [datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
             str(uuid.uuid4()).split('-')[0]]
        )
    output_dir, = accelerate.utils.broadcast_object_list([os.path.join(args.output_dir, args.exp_name)], 0)
    return output_dir


def unwrap_model(model):
    model = model._orig_mod if hasattr(model, '_orig_mod') else model # For compiled models
    model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model # For DDP models
    return model


"""
Accelerator & logger
"""
def get_accelerator_logger(exp_cfg):
    logger = logging.getLogger(__name__)

    # Accelerator setting
    project_config = accelerate.utils.ProjectConfiguration(save_on_each_node=True)
    dataloader_config = accelerate.DataLoaderConfiguration(non_blocking=True)
    kwargs_handlers = [
        accelerate.DistributedDataParallelKwargs(comm_hook=accelerate.DDPCommunicationHookType.NO,
                                                 gradient_as_bucket_view=True, broadcast_buffers=False),
        accelerate.InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=5)),
    ]

    # Accelerator
    accelerator = accelerate.Accelerator(project_config=project_config,
                                         dataloader_config=dataloader_config,
                                         kwargs_handlers=kwargs_handlers)

    # Prepare output dir & save configurations
    output_dir = get_output_path(exp_cfg)

    if os.path.exists(output_dir):
        if exp_cfg.resume_if_exists:
            logger.info(f'Output dir {output_dir} already exists. Resuming training.')
            exp_cfg.resume = output_dir
        else:
            raise ValueError(f'Output dir {output_dir} already exists. Please change exp_name or set resume_if_exists to True.')
    accelerator.wait_for_everyone()

    if not os.path.exists(output_dir):
        if accelerator.is_main_process:
            os.makedirs(output_dir)
    accelerator.wait_for_everyone()

    # Log setting
    os.makedirs(os.path.join(output_dir, f'rank_{accelerator.process_index}'), exist_ok=True)
    if accelerator.is_main_process:
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

    if accelerator.num_processes > 1:
        file_formatter = logging.Formatter('[%(asctime)s] ' + 'Rank - ' + \
                                           str(accelerator.process_index) + ' %(levelname)s - %(message)s')
    else:
        file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(output_dir, f'rank_{accelerator.process_index}', 'logs.txt'))
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # World setting
    logger.info(f'Torch version: {torch.__version__}')
    logger.info(f'World size: {accelerator.num_processes}')
    logger.info(f'Rank: {accelerator.process_index}    Local rank: {accelerator.local_process_index}    Device: {accelerator.device}')
    if accelerator.device.type == 'cuda':
        logger.info(f'Device name: {torch.cuda.get_device_name(accelerator.device)}')

    return accelerator, logger, output_dir


def end_training(logger, accelerator):
    accelerator.end_training()


"""
Others
"""
class CheckpointManager:
    def __init__(self, logger, accelerator, output_dir):
        self.logger = logger
        self.accelerator = accelerator
        self.output_dir = output_dir

    def _save_meta(self, path, data):
        with open(os.path.join(path, 'meta.yaml'), 'w') as file:
            yaml.dump(data, file)

    def _load_meta(self, path):
        with open(os.path.join(path, 'meta.yaml'), 'r') as file:
            data = yaml.full_load(file)
        return data

    def save(self, iters, metrics={}):
        # Save the new checkpoint
        if self.accelerator.is_main_process:
            self.accelerator.save_state(output_dir=self.output_dir, safe_serialization=False)
            meta = {'iter': iters, 'metric': metrics}
            self._save_meta(self.output_dir, meta)

    def load(self, path):
        # Load the checkpoint
        self.accelerator.load_state(path)
        meta = self._load_meta(path)
        return meta['iter']


def update_buffer(buffer, update, max=50):
    if isinstance(buffer, dict):
        for k, v in update.items():
            if k not in buffer:
                buffer[k] = []
            buffer[k].append(v)
            if max is not None and len(buffer[k]) > max:
                buffer[k].pop(0)
    elif isinstance(buffer, list):
        buffer.append(update)
        if max is not None and len(buffer) > max:
            buffer.pop(0)
    else:
        raise ValueError


def average_buffer(buffer):
    if isinstance(buffer, dict):
        return {k: sum(v) / len(v) for k, v in buffer.items()}
    elif isinstance(buffer, list):
        return sum(buffer) / len(buffer)
    else:
        raise ValueError


def reduce_buffer(accelerator, buffer, to_item=True, reduction='mean'):
    results = accelerator.reduce(average_buffer(buffer), reduction=reduction)
    if to_item:
        return {k: v.item() for k, v in results.items()} if isinstance(results, dict) else results.item()
    else:
        return results


def gather_buffer(accelerator, buffer):
    out = {}
    for key, buf_list in buffer.items():
        out[key] = {
            k: accelerator.gather(torch.cat([b[k] for b in buf_list], dim=0).to(accelerator.device)).cpu()
            for k in buf_list[0]
        }
    return out