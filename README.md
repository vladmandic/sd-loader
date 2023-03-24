# sd-loader

Stable Diffusion model loader tool with benchmark:

- comparing **file** and **stream** loading methods
- for both `torch` checkpoint and `safetensors` formats

1. Install requirements
2. Download models  
   Example: download both `ckpt` and `safetensor` from <https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main>
3. Update paths and model names in `bench.py`
4. Run

Example output:

```log
10:05:45-290401 torch: 2.0.0+cu118 cuda: 11.8 cudnn: 8700 gpu: NVIDIA GeForce RTX 3060 capability: (8, 6)
10:05:45-291374 start tests using file method
10:05:45-291844 load model v2-1_768-ema-pruned.ckpt using file method pass 1 of 3 start
10:05:45-586631 0.006 load config: v2-inference-768-v.yaml (cpu: 0.861 gpu: 1.002 gc: 0.289)
10:05:54-967983 9.292 create model: ldm.models.diffusion.ddpm.LatentDiffusion (cpu: 5.864 gpu: 1.002 gc: 0.088)
10:06:05-383291 10.32 load weights: /mnt/d/Models/v2-1_768-ema-pruned.ckpt (cpu: 10.695 gpu: 1.002 gc: 0.095)
10:06:05-485051 0.0 state dict: 1242 (cpu: 10.695 gpu: 1.002 gc: 0.101)
10:06:05-935847 0.356 apply weigths (cpu: 10.695 gpu: 1.002 gc: 0.094)
10:06:06-370971 0.337 model eval (cpu: 8.45 gpu: 1.002 gc: 0.097)
10:06:07-256858 0.797 model move (cpu: 8.045 gpu: 3.532 gc: 0.088)
10:06:07-372472 0.001 model unload (cpu: 8.045 gpu: 1.002 gc: 0.114)
10:06:07-378629 load model v2-1_768-ema-pruned.ckpt using file method pass 1 of 3 in 22.087 seconds
```
