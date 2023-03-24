#!/bin/env python

# system imports
import io
import os
import gc
import time
import warnings
import argparse

# library imports
import psutil
import torch
import safetensors
import pytorch_lightning # pylint: disable=unused-import
from omegaconf import OmegaConf
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install
from rich.console import Console

# modules imports: this is unmodified sd code
import ldm.util


# parse arguments
parser = argparse.ArgumentParser(description = 'sd-loader')
parser.add_argument('--model', type=str, default=None, required=True, help='model name')
parser.add_argument('--method', type=str, default=None, choices=['file', 'stream'], required=True, help='load method')
parser.add_argument('--repeats', type=int, default=3, required=False, help='number of repeats')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], required=False, help='load initial target')
parser.add_argument('--config', type=str, default='v2-inference-768-v.yaml', required=False, help='model config')
parser.add_argument('--dtype', type=str, default='fp16', choices=['fp32', 'fp16', 'bf16'], required=False, help='target dtype')
args = parser.parse_args()


class Logger:
    def __init__(self):
        self.t = time.perf_counter()
        self.console = Console(log_time=True, log_time_format='%H:%M:%S-%f')
        pretty_install(console=self.console)
        traceback_install(console=self.console, extra_lines=1, width=self.console.width, word_wrap=False, indent_guides=False, suppress=[torch])
        warnings.filterwarnings(action='ignore', category=UserWarning)
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        self.process = psutil.Process(os.getpid())

    def gb(self, val: float):
        return round(val / 1024 / 1024 / 1024, 3)

    def log(self, msg: str):
        self.console.log(msg)

    def start(self):
        self.t = time.perf_counter()

    def trace(self, msg: str):
        cpu = self.gb(self.process.memory_info().rss)
        gpu_info = torch.cuda.mem_get_info()
        gpu = self.gb(gpu_info[1] - gpu_info[0])
        t = time.perf_counter()
        self.console.log(f'{round(t - self.t, 3)} {msg} (cpu: {cpu} gpu: {gpu})')
        self.t = t
logger = Logger()


def garbage_collect():
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    logger.trace('garbage collect')


def load_model():
    logger.start()

    model_config = OmegaConf.load(args.config)
    model = ldm.util.instantiate_from_config(model_config.model)
    logger.trace(f'create model: {model_config.model["target"]} from {{args.config}}')

    _, extension = os.path.splitext(args.model)
    if args.method == 'stream':
        with open(args.model,'rb') as f:
            if extension.lower()=='.safetensors':
                buffer = f.read()
                weights = safetensors.torch.load(buffer)
            else:
                buffer = io.BytesIO(f.read())
                weights = torch.load(buffer, map_location=args.device)
    elif args.method == 'file':
        if extension.lower()=='.safetensors':
            weights = safetensors.torch.load_file(args.model, device=args.device)
        else:
            weights = torch.load(args.model, map_location=args.device)
    logger.trace(f'load weights: {args.model}')

    state_dict = weights.pop('state_dict', weights)
    state_dict.pop('state_dict', None)

    model.load_state_dict(state_dict, strict=False)
    del weights # unload weigts since they were applied to model
    logger.trace(f'apply weigths to dict: {len(state_dict)})')

    dtype = torch.float32 if args.dtype == 'fp32' else torch.float16 if args.dtype == 'fp16' else torch.bfloat16
    model.to(dtype)
    model.eval()
    logger.trace('model eval')

    model.to('cuda')
    logger.trace('model move')

    return model


if __name__ == '__main__':
    logger.log(f'torch: {torch.__version__} cuda: {torch.version.cuda} cudnn: {torch.backends.cudnn.version()} gpu: {torch.cuda.get_device_name()} capability: {torch.cuda.get_device_capability()}')
    logger.log(f'options: {vars(args)}')
    for i in range(args.repeats):
        t0 = time.perf_counter()
        sd = load_model()
        logger.log(f'load model pass {i + 1}: {round(time.perf_counter() - t0, 3)} seconds')
        del sd # unload model
        garbage_collect()
