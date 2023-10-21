"""
notice that the unet is different from the original unet which takes x and t as input
here we fix the t in order to get the partial derivative regarding to x conveniently
line 384 in denoising_diffusion_pytorch.py
"""

import matplotlib.pyplot as plt
import math
import copy
from pathlib import Path
from random import random
import random
import numpy as np
from utils import *

import torch
from torch import nn, einsum

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from tqdm import trange

from accelerate import Accelerator

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

import tyro
from config import Args

# load configuration
args = tyro.cli(Args)

# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" +
      (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

# Setting reproducibility
SEED = args.SEED
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def grad_desc(model, x, noise, args=args):
    # parameters
    gd_lr = args.gd_lr
    gd_epochs = args.gd_epochs

    x_copy = x.clone()
    optimizer = torch.optim.SGD([x_copy], lr=gd_lr)
    criterion = torch.nn.MSELoss()

    for epoch in trange(gd_epochs):
        optimizer.zero_grad()

        output = model(x_copy)
        loss = criterion(output, noise)

        loss.backward()
        optimizer.step()
    
    return x_copy


# initialize x for gradient descent -->
# 1. random noise  2. generated new images  3. interpolation of generated images and nearest neighbot in train set  4. test images
def initialize(t, noise, lam, args=args):
    choice = args.choice
    gen_path = args.gen_path
    real_path = args.real_path
    test_path = args.test_path

    assert choice in ['noise', 'gen', 'intp', 'test']

    if choice == 'noise':
        return torch.randn_like(noise).to(device)
    elif choice == 'gen':
        gen = np.load(gen_path)
        gen = normalize(gen)  # normalize to [-1, 1]
        return q_sample(torch.from_numpy(gen).to(device), t, noise)  # forward process
    elif choice == 'intp':
        gen = np.load(gen_path)
        real = np.load(real_path)
        gen = normalize(gen)  # normalize to [-1, 1]
        real = normalize(real)  # normalize to [-1, 1]
        intp = lam * gen + (1 - lam) * real
        return q_sample(torch.from_numpy(intp).to(device), t, noise)  # forward process
    elif choice == 'test':
        test = np.load(test_path)
        test = normalize(test)  # normalize to [-1, 1]
        return q_sample(torch.from_numpy(test).to(device), t, noise)  # forward process


# main
def main():
    save_fig_path = args.save_fig_path

    # load diffusion model
    model, diffusion, trainer = load_model(args=args)
    batch_size = args.batch_size
    c = args.channels
    h = w = args.image_size

    # t
    time = args.time
    t = torch.full((batch_size,), time, device=device).long()

    # noise
    noise = torch.randn(batch_size, c, h, w).to(device)

    # initialize
    x_input = initialize(t, noise, args=args)

    # loss before gradient descent
    with torch.no_grad():
        loss_before = (model(x_input) - noise).norm(p=2)

    x_input.requires_grad=True
    noise.requires_grad = True
    

    # gradient descent
    gd_min = grad_desc(model, x_input, noise)

    gd_min_back = (gd_min - torch.sqrt(1 - alphas_cumprod[time])) / torch.sqrt(alphas_cumprod[time])
    gd_min_clamp = torch.clamp(gd_min_back, -1, 1)
    result_np = gd_min_clamp.cpu().detach().numpy()  # shape [batch_size, c, h, w]
    '''
    x_ori = x_np[pic_index]

    dist = []

    for i in range(50):
        image = Image.open(f'mnist_50/image_{i}.png')
        image = np.array(image).astype(np.float32)
        image /= 255.0
        img = (result_np + 1) / 2.0

        dist.append(np.linalg.norm((img - image), ord=2))

    print(min(dist))
    train_near = Image.open(f'mnist_50/image_{dist.index(min(dist))}.png')

    fig, axs = plt.subplots(1, 3, figsize=(6, 6))
    axs = axs.flatten()

    axs[2].imshow(result_np, cmap='gray')
    axs[2].axis('off')
    axs[2].set_title('after gd')

    axs[1].imshow(x_ori[0], cmap='gray')
    axs[1].axis('off')
    axs[1].set_title('before gd')

    axs[0].imshow(train_near, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('nearest(to after gd) train image')

    # save
    plt.savefig(save_fig_path)

    plt.tight_layout()
    plt.show()
    '''
    loss_after = (model(gd_min) - noise).norm(p=2)
    print('loss before gd: ', loss_before.item())
    print('loss after gd: ', loss_after.item())


