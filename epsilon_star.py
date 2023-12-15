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

def epsilon_star_batch(x:torch.Tensor, t:torch.Tensor, dataloader:np.ndarray, device):
    b_size=5000  # if cuda OOM, decrease b_size!
    b, c, h, w = x.shape  # take [256, 3, 32, 32] as an example
    num_train = dataloader.shape[0]  # take 5k as an example
    num_batches = (num_train + b_size - 1) // b_size

    time = t[0].item()  # int number
    alpha_bar = alphas_cumprod[time]  # float number
    # *********************************
    numerator = torch.zeros(b, c, h, w).to(device)  # [256, 3, 32, 32]
    denominator = torch.zeros(b).to(device)
    # deal with batch
    max_norm = -torch.inf * torch.ones(b).to(device)
    for i in range(num_batches):
        start_idx = i * b_size
        end_idx = min((i + 1) * b_size, num_train)
        train_data = torch.from_numpy(dataloader[start_idx:end_idx]).to(device)  # a batch of train data  [batch_size, 3, 32, 32]
        batch_train_data = train_data.repeat(b, 1, 1, 1).reshape(b, train_data.shape[0], c, h, w)  # [256, batch_size, 3, 32, 32]
        x_minus_x0 = x[:, None, :, :, :] - torch.sqrt(alpha_bar) * batch_train_data  # [256, batch_size, 3, 32, 32]
        norm = x_minus_x0.norm(p=2, dim=(2, 3, 4)) ** 2 / (-2 * (1 - alpha_bar))  # [256, batch_size]

        # max of norm
        max_norm = torch.max(max_norm, norm.max(dim=1)[0])

    for i in range(num_batches):
        start_idx = i * b_size
        end_idx = min((i + 1) * b_size, num_train)
        train_data = torch.from_numpy(dataloader[start_idx:end_idx]).to(device)  # a batch of train data  [batch_size, 3, 32, 32]
        batch_train_data = train_data.repeat(b, 1, 1, 1).reshape(b, train_data.shape[0], c, h, w)  # [256, batch_size, 3, 32, 32]
        x_minus_x0 = x[:, None, :, :, :] - torch.sqrt(alpha_bar) * batch_train_data  # [256, batch_size, 3, 32, 32]
        norm = x_minus_x0.norm(p=2, dim=(2, 3, 4)) ** 2 / (-2 * (1 - alpha_bar))  # [256, batch_size]

        # softmax
        # numerator -- a tensor: \sum [exp()*xi]
        numerator += (torch.exp(norm - max_norm[:, None])[:, :, None, None, None] * train_data[None, :, :, :, :]).sum(dim=1)  # [256, 3, 32, 32]

        # denominator -- a real number \sum exp()
        denominator += torch.sum(torch.exp(norm - max_norm[:, None]), dim=1)  # [256, ]

    weighted_x = numerator / denominator[:, None, None, None]    

    return x / torch.sqrt(1 - alpha_bar) - torch.sqrt(alpha_bar) / torch.sqrt(1 - alpha_bar) * weighted_x

def eps_star_fix_t(x:torch.Tensor):
    b_size=5000  # if cuda OOM, decrease b_size!
    dataloader = np.load(args.train_path)
    time = args.time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b, c, h, w = x.shape  # take [256, 3, 32, 32] as an example
    num_train = dataloader.shape[0]  # take 5k as an example
    num_batches = (num_train + b_size - 1) // b_size

    alpha_bar = alphas_cumprod[time]  # float number
    # *********************************
    numerator = torch.zeros(b, c, h, w).to(device)  # [256, 3, 32, 32]
    denominator = torch.zeros(b).to(device)
    # deal with batch
    max_norm = -torch.inf * torch.ones(b).to(device)
    for i in range(num_batches):
        start_idx = i * b_size
        end_idx = min((i + 1) * b_size, num_train)
        train_data = torch.from_numpy(dataloader[start_idx:end_idx]).to(device)  # a batch of train data  [batch_size, 3, 32, 32]
        batch_train_data = train_data.repeat(b, 1, 1, 1).reshape(b, train_data.shape[0], c, h, w)  # [256, batch_size, 3, 32, 32]
        x_minus_x0 = x[:, None, :, :, :] - torch.sqrt(alpha_bar) * batch_train_data  # [256, batch_size, 3, 32, 32]
        norm = x_minus_x0.norm(p=2, dim=(2, 3, 4)) ** 2 / (-2 * (1 - alpha_bar))  # [256, batch_size]

        # max of norm
        max_norm = torch.max(max_norm, norm.max(dim=1)[0])

    for i in range(num_batches):
        start_idx = i * b_size
        end_idx = min((i + 1) * b_size, num_train)
        train_data = torch.from_numpy(dataloader[start_idx:end_idx]).to(device)  # a batch of train data  [batch_size, 3, 32, 32]
        batch_train_data = train_data.repeat(b, 1, 1, 1).reshape(b, train_data.shape[0], c, h, w)  # [256, batch_size, 3, 32, 32]
        x_minus_x0 = x[:, None, :, :, :] - torch.sqrt(alpha_bar) * batch_train_data  # [256, batch_size, 3, 32, 32]
        norm = x_minus_x0.norm(p=2, dim=(2, 3, 4)) ** 2 / (-2 * (1 - alpha_bar))  # [256, batch_size]

        # softmax
        # numerator -- a tensor: \sum [exp()*xi]
        numerator += (torch.exp(norm - max_norm[:, None])[:, :, None, None, None] * train_data[None, :, :, :, :]).sum(dim=1)  # [256, 3, 32, 32]

        # denominator -- a real number \sum exp()
        denominator += torch.sum(torch.exp(norm - max_norm[:, None]), dim=1)  # [256, ]

    weighted_x = numerator / denominator[:, None, None, None]    

    return x / torch.sqrt(1 - alpha_bar) - torch.sqrt(alpha_bar) / torch.sqrt(1 - alpha_bar) * weighted_x

def eps_star_jacobian(x:torch.Tensor, t:torch.Tensor, dataloader:np.ndarray):
    '''x have to be one image with size [c, h, w]'''
    time = t[0].item()
    alpha_t_bar = alphas_cumprod[time].item()
    sqrt_alpha_t_bar = sqrt_alphas_cumprod[time].item()

    # coefficient: exp(-|| ||/2(1-bar_alpha))
    x_0 = torch.tensor(dataloader).to('cuda')
    coefficients = F.softmax(torch.norm(x[None, :] - sqrt_alpha_t_bar*x_0, p=2, dim=(1, 2, 3))/(-2 * (1-alpha_t_bar)))

    # change image into vector
    x0_vector = x_0.view(x_0.shape[0], -1)
    x_vector = x.view(-1)
    x_minus_x0_vector = coefficients[:, None] * (x_vector[None, :] - sqrt_alpha_t_bar * x0_vector)
    x0_vector = coefficients[:, None] * x0_vector

    # calculate
    result = torch.matmul(x0_vector.T, x_minus_x0_vector)
    
    return result


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
    eps_star = epsilon_star_batch(x_input, t, loader, device=device)


if __name__ == '__main__':
    main()
    