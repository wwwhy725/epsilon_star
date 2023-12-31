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
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import numpy as np

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torch.utils.data import Subset

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from tqdm import trange

from accelerate import Accelerator

from utils import *


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

    optimizer = torch.optim.SGD([x], lr=gd_lr)
    criterion = torch.nn.MSELoss()

    for epoch in trange(gd_epochs):
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, noise)

        loss.backward()
        optimizer.step()
    
    return x


# initialize x for gradient descent -->
# 1. random noise  2. generated new images  3. interpolation of generated images and nearest neighbot in train set  4. test images
def initialize(t, noise, lam=0.5, args=args):
    choice = args.choice
    gen_path = args.gen_path
    real_path = args.real_path
    test_path = args.test_path

    assert choice in ['noise', 'gen', 'intp', 'test']

    if choice == 'noise':
        noise_np = noise.cpu().detach().numpy()
        return torch.randn_like(noise).to(device)  # !!!! 这块还有点小问题，用到再调
    elif choice == 'gen':
        gen = np.load(gen_path)
        gen = normalize(gen)  # normalize to [-1, 1]
        return gen, q_sample(torch.from_numpy(gen).to(device), t, noise).to(torch.float32)  # forward process
    elif choice == 'intp':
        gen = np.load(gen_path)
        real = np.load(real_path)
        gen = normalize(gen)  # normalize to [-1, 1]
        real = normalize(real)  # normalize to [-1, 1]
        intp = lam * gen + (1 - lam) * real
        return intp, q_sample(torch.from_numpy(intp).to(device), t, noise).to(torch.float32)  # forward process
    elif choice == 'test':
        test = np.load(test_path)
        test = normalize(test)  # normalize to [-1, 1]
        return test, q_sample(torch.from_numpy(test).to(device), t, noise).to(torch.float32)  # forward process


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
    x_ori, x_input = initialize(t, noise, args=args)

    x_input.requires_grad = True
    noise.requires_grad = True

    # loss before gradient descent
    with torch.no_grad():
        loss_before = (model(x_input) - noise).norm(p=2, dim=(1, 2, 3)).mean()
    
    # gradient descent
    gd_min = grad_desc(model, x_input, noise)

    gd_min_back = (gd_min - torch.sqrt(1 - alphas_cumprod[time])) / torch.sqrt(alphas_cumprod[time])
    gd_min_clamp = torch.clamp(gd_min_back, -1, 1)
    result_np = gd_min_clamp.cpu().detach().numpy()  # shape [batch_size, c, h, w]
    # print(result_np.max(), result_np.min())
    result_np = (result_np + 1) / 2.0
    x_ori =  (x_ori + 1) / 2.0

    fig, axs = plt.subplots(2, batch_size, figsize=(15, 15)) # tbd  这里会不会有点太多了
    if args.channels == 3:
        for i in range(batch_size):
            axs[0, i].imshow(result_np[i].transpose(1, 2, 0))  # after gradient descent -- a batch of images
            axs[0, i].axis('off')
            axs[1, i].imshow(x_ori[i].transpose(1, 2, 0))  # before gradient descent -- a batch of images
            axs[1, i].axis('off')
    elif args.channels == 1:
        for i in range(batch_size):
            axs[0, i].imshow(result_np[i].transpose(1, 2, 0), cmap='gray')  # after gradient descent -- a batch of images
            axs[0, i].axis('off')
            axs[1, i].imshow(x_ori[i].transpose(1, 2, 0), cmap='gray')  # before gradient descent -- a batch of images
            axs[1, i].axis('off')
    else:
        raise ValueError('channels have to be 1 or 3!')
    plt.tight_layout()
    # save
    plt.savefig(save_fig_path)
    plt.show()
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
    loss_after = (model(gd_min) - noise).norm(p=2, dim=(1, 2, 3)).mean()
    print('loss before gd: ', loss_before.item())
    print('loss after gd: ', loss_after.item())


if __name__ == '__main__':
    main()

