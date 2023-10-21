from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 3,
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,           # number of steps
    objective = 'pred_noise',
    beta_schedule = 'linear'
)

trainer = Trainer(
    diffusion,
    'cifar_subset',
    train_batch_size = 1024,
    train_lr = 1e-4,
    train_num_steps = 30000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    num_samples = 256,
    results_folder = './results_cifar10subset_pen_1e-4weight',
    amp = False,                       # turn on mixed precision
    calculate_fid = False              # whether to calculate fid during training
)

trainer.train()