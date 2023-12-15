import math
import copy
import os
from pathlib import Path
from random import random
import random
from functools import partial
import numpy as np

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm


from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.version import __version__

from config import Args
import tyro

# load configuration
args = tyro.cli(Args)

# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# alphas
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)

betas = linear_beta_schedule(1000).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def q_sample(x_start, t, noise=None):
    noise = default(noise, lambda: torch.randn_like(x_start))

    return (
            extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channel,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channel = channel
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        if self.channel == 1:
            self.transform = T.Compose([
                T.Lambda(maybe_convert_fn),
                T.Resize(image_size),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
        elif self.channel == 3:
            self.transform = T.Compose([
                T.Lambda(maybe_convert_fn),
                T.Resize(image_size),
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            raise ValueError('channel have to be 1 or 3!')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# prepare the dataloader
def get_trainloader(image_size, channel, batch_size, train_path):
    ds = Dataset(train_path, image_size=image_size, channel=channel, augment_horizontal_flip=False, convert_image_to = None)
    loader = DataLoader(ds, batch_size = batch_size, shuffle = False)

    return loader

# load diffusion model
def load_model(args=args):
    if args.channels == 1:
        model = Unet(
        dim = args.dim,
        dim_mults = (1, 2),
        channels = args.channels,
        full_attn = (False, True),
        flash_attn = True
        )
    elif args.channels == 3:
        model = Unet(
        dim = args.dim,
        dim_mults = (1, 2, 4, 8),
        channels = args.channels,
        flash_attn = True
        )
    else:
        raise ValueError('image channels have to be either 1 or 3!')

    diffusion = GaussianDiffusion(
        model,
        image_size = args.image_size,
        timesteps = 1000,           # number of steps
        sampling_timesteps=args.ddim_timesteps,
        objective = 'pred_noise',
        beta_schedule = 'linear'
    )

    trainer = Trainer(
        diffusion,
        args.folder,                         # training set folder -- .png
        train_batch_size = args.train_batch_size,
        train_lr = args.train_lr,
        train_num_steps = args.train_num_steps,    # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        num_samples = 256,
        results_folder = args.results_folder,      # results
        amp = False,                       # turn on mixed precision
        calculate_fid = args.calculate_fid,  # whether to calculate fid during training
        num_fid_samples=args.num_fid_samples
    )

    trainer.load(args.ckpt)

    return model, diffusion, trainer

def save_numpy_image(image, path):
    # normalize image to 0-255
    image = image - image.min()
    image /= image.max()
    image *= 255
    image = image.astype(np.uint8)
    # save image
    image = Image.fromarray(image)
    image.save(path)

def normalize(arr: np.ndarray):
    # [0, 255] to [-1, 1]
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
        arr = arr * 2 - 1.0
    # [0, 1] to [-1, 1]
    elif arr.dtype == np.float32:
        if arr.min() == 0.0:
            arr = arr * 2 - 1.0
    
    return arr

def normalize_to_255(image):
    # normalize image to 0-255
    image = image - image.min()
    image /= image.max()
    image *= 255
    image = image.astype(np.uint8)
    return image