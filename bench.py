import io
import os
import gc
import time
import warnings
import torch
import safetensors
import ldm.util
import psutil
import pytorch_lightning # pylint: disable=unused-import
from omegaconf import OmegaConf
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install
from rich.console import Console


repeats = 3
checkpoint_folder = '/mnt/d/Models' # update path with your location
checkpoint_files = [ # download files from (https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main)
    'v2-1_768-ema-pruned.ckpt',
    'v2-1_768-ema-pruned.safetensors'
]
checkpoint_config = 'v2-inference-768-v.yaml'


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
        gc_start = time.perf_counter()
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        cpu = self.gb(self.process.memory_info().rss)
        gpu_info = torch.cuda.mem_get_info()
        gpu = self.gb(gpu_info[1] - gpu_info[0])
        gc_time = round(time.perf_counter() - gc_start, 3)
        self.console.log(f'{round(gc_start - self.t, 3)} {msg} (cpu: {cpu} gpu: {gpu} gc: {gc_time})')
        self.t = time.perf_counter()
logger = Logger()


def load_weights(checkpoint_file, load_method):
    _, extension = os.path.splitext(checkpoint_file)
    if load_method == 'stream':
        with open(checkpoint_file,'rb') as f:
            if extension.lower()=='.safetensors':
                buffer = f.read()
                weights = safetensors.torch.load(buffer)
            else:
                buffer = io.BytesIO(f.read())
                weights = torch.load(buffer, map_location='cpu')
    elif load_method == 'file':
        if extension.lower()=='.safetensors':
            weights = safetensors.torch.load_file(checkpoint_file, device='cpu')
        else:
            weights = torch.load(checkpoint_file, map_location='cpu')
    return weights


def load_model(checkpoint_file, load_method):
    logger.start()

    model_config = OmegaConf.load(checkpoint_config)
    logger.trace(f'load config: {checkpoint_config}')
    model = ldm.util.instantiate_from_config(model_config.model)
    logger.trace(f'create model: {model_config.model["target"]}')

    weights = load_weights(checkpoint_file, load_method)
    logger.trace(f'load weights: {checkpoint_file}')

    state_dict = weights.pop('state_dict', weights)
    state_dict.pop('state_dict', None)
    logger.trace(f'state dict: {len(state_dict)}')

    model.load_state_dict(state_dict, strict=False)
    weights = None # unload weigts since they were applied to model
    logger.trace('apply weigths')

    model.to(torch.float16)
    model.eval()
    logger.trace('model eval')

    model.to('cuda')
    logger.trace('model move')

    model = None
    logger.trace('model unload')
    return model # pointless since its unloaded


if __name__ == '__main__':
    logger.log(f'torch: {torch.__version__} cuda: {torch.version.cuda} cudnn: {torch.backends.cudnn.version()} gpu: {torch.cuda.get_device_name()} capability: {torch.cuda.get_device_capability()}')
    for fn in checkpoint_files:
        for method in ['file', 'stream']:
            logger.log(f'start tests using {method} method')
            for i in range(repeats):
                t0 = time.perf_counter()
                logger.log(f'load model {fn} using {method} method pass {i + 1} of {repeats} start')
                sd = load_model(os.path.join(checkpoint_folder, fn), method)
                logger.log(f'load model {fn} using {method} method pass {i + 1} of {repeats} in {round(time.perf_counter() - t0, 3)} seconds')
