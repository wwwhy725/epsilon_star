import math
import copy
from pathlib import Path
from random import random
import random
from functools import partial
from collections import namedtuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import *

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


# Setting reproducibility
SEED = args.SEED
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" +
    (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

# empirical min
def epsilon_star(x:torch.Tensor, t:torch.Tensor, dataloader:np.ndarray, device):
    b, c, w, h = x.shape  # take [256, 3, 32, 32] as an example

    time = t[0].item()
    alpha_bar = alphas_cumprod[time]
    train_data = torch.from_numpy(dataloader).to(device)
    num_train = train_data.shape[0]  # take 5k as an example
    batch_train_data = train_data.repeat(b, 1, 1, 1).reshape(b, num_train, c, w, h)  # [256, 5k, 3, 32, 32]

    x_minus_x0 = x[:, None, :, :, :] - torch.sqrt(alpha_bar) * batch_train_data  # [256, 5k, 3, 32, 32]
    norm = x_minus_x0.norm(p=2, dim=(2, 3, 4)) ** 2 / (-2 * (1 - alpha_bar))  # [256, 5k]

    softmax = F.softmax(norm, dim=1)
    weighted_x = (softmax[:, :, None, None, None] * train_data[None, :, :, :, :]).sum(dim=1)  # [256, 3, 32, 32]

    return x / torch.sqrt(1 - alpha_bar) - torch.sqrt(alpha_bar) / torch.sqrt(1 - alpha_bar) * weighted_x


def main():  # 一个拿到epsilon_star的模板
    # load data
    batch_data_path = args.gen_path
    train_data_path = args.real_path
    batch_size = args.batch_size
    x_np = np.load(batch_data_path)
    loader = np.load(train_data_path)

    # normalize to [-1, 1]
    x_np = normalize(x_np)
    
    # to tensor
    x = torch.from_numpy(x_np).to(device)

    # noise
    noise = None
    noise = default(noise, lambda: torch.randn_like(x))  # 这儿的noise通常要fix！代码可能要改改

    # t
    time = args.time
    t = torch.full((batch_size,), time, device=device).long()

    # forward process on x
    x_input = q_sample(x, t, noise).to(torch.float32)

    # epsilon*
    eps_star = epsilon_star(x_input, t, loader, device=device)


if __name__ == '__main__':
    main()
    