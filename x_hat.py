import matplotlib.pyplot as plt
import math
import copy
from pathlib import Path
from random import random
import random
import numpy as np

from PIL import Image
from tqdm.auto import tqdm

from utils import *
from epsilon_star import epsilon_star


# Setting reproducibility
SEED = args.SEED
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" +
      (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))


def weighted_x0(x, t, epsilon, args=args):
    b, c, h, w = x.shape
    
    loader = np.load(args.train_path)

    # get the model_out
    if args.mode == 'trained':
        model_out = epsilon(x, t, x_self_cond = None)
    elif args.mode == 'empirical':
        model_out = epsilon(x, t, dataloader=loader, device=device)
    else:
        raise ValueError('mode have to be trained or empirical!')

    # alphas
    alpha_expand = alphas_cumprod[t].unsqueeze(1).unsqueeze(2).unsqueeze(3).to(device)
    
    # calculate weighted x0
    result = x / torch.sqrt(alpha_expand) - (torch.sqrt(1 - alpha_expand) / torch.sqrt(alpha_expand)) * model_out
    # print('min, max = ', torch.min(result), torch.max(result))

    # visualization
    array = result.cpu().detach().numpy()

    # 调整数组的形状以适应plt.imshow函数的要求
    array = array.reshape(b, h, w)

    '''
    fig, axs = plt.subplots(1, b, figsize=(10, 6))
    axs = axs.flatten()

    for i in range(b):
        axs[i].imshow(array[i], cmap='gray')
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
    '''
    return result

def main():
    # convert x_start to x
    # noise = default(noise, lambda: torch.randn_like(x_start))
    # x = q_sample(x_start = x_start, t = t, noise = noise).to(torch.float32)
    return 0


if __name__ == '__main__':
    main()