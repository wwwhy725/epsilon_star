from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import logging
from utils import *


logger = logging.getLogger(__name__)

# to get the closest factors of a   e.g.  12 = 3 * 4 = 2 * 6  what I want is 3,4 rather than 2,6
def closest_factors(a):
    factors = []
    for i in range(1, int(a ** 0.5) + 1):
        if a % i == 0:
            factors.append((i, a // i))
    closest_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
    return closest_factor

'''use diffusion model to sample'''
def sampling(args=args):
    _, diffusion, _ = load_model(args)

    # sampling
    samples = []
    for _ in range(args.sample_num):
        gen = diffusion.sample(batch_size=args.batch_size)
        samples.append(gen)
    samples = torch.concat(samples)
    print(samples.shape)
    print(samples.max(), samples.min())
    if samples.min() >= 0.0:  # normalize to [-1, 1]
        samples = samples * 2 - 1.0

    # save
    np.save(args.save_np_path, samples.cpu().detach().numpy())

    return samples

def knn_ratio_metric(gen_img:torch.Tensor, train_img:torch.Tensor, n=50, alpha=0.5):
    """
    alpha: float, hyper-parameter
    n: int, n nearest neighbour
    return: nearest_dist / (alpha/n * \sum_{i=1}^n dist(x, x_i)), and the index of most memorized training image
    """
    b, c, h, w = train_img.shape
    flatten_shape = c * h * w
    dif = (gen_img[None, :] - train_img).reshape((b, flatten_shape))
    # dif_norm = torch.sqrt(dif.norm(p=2, dim=-1) ** 2 / flatten_shape)
    dif_norm = torch.norm(dif, p=2, dim=-1)
    top_val, top_idx = torch.topk(dif_norm, k=n, largest=False)
    
    ratio = top_val[0] / (alpha * (torch.sum(top_val)) / n)

    return ratio.item(), top_idx[0].item()

def main():
    # Setup logging
    log_name = args.log_name
    logging.basicConfig(filename=log_name,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

    logger.info('****************************')
    # logger.info('training with 1e-4 * pen')
    # logger.info('timesteps sampling')

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sample images and save as .npy
    logger.info('ddpm sampling......')
    samples = sampling(sample_num=args.sample_num, ckpt=args.ckpt, save_path=args.gen_all_path)
    logger.info('finished!')
    
    # load train data
    train_np = np.load(args.train_path)
    train_imgs = torch.from_numpy(train_np).to(device)

    # calculate memorization
    memo_idx = {}
    threshold = 1
    # gen_memo_info = []

    logger.info('memorization metric: knn ratio')
    logger.info(f'threshold={threshold}')

    results = []
    for index, gen_img in tqdm(enumerate(samples), total=len(samples), desc="Processing"):
        ratio, top_idx = knn_ratio_metric(gen_img, train_imgs)
        results.append([ratio, index, top_idx])
        if ratio < threshold:
            # memo_idx.append([index, top_idx])  # 存下来gen-train的索引
            memo_idx[top_idx] = index  # 存下来 train_idx : gen_idx
    
    count = len(memo_idx)  # 不算重复的，总共的memorize数量

    if count > 0:
        logger.info(f'number of imgs memorized: {count}')
        for key in memo_idx:
            logger.info(f'train index is {key}, gen indx is {memo_idx[key]}')

    else:
        logger.info('no image memorized!')

    # visualization
    results = np.array(results)
    gen_np = samples.cpu().detach().numpy()
    gen_np = (gen_np + 1) / 2.0  # normalize to [0, 1]

    idx = results[results[:, 0].argsort()][:, 1:]
    canvas = np.empty((args.sample_num * args.batch_size, args.image_size, 2 * args.image_size, args.channels))
    row, col = closest_factors(args.sample_num * args.batch_size)
    for i in range(args.sample_num * args.batch_size):
        cat = np.concatenate((samples[int(idx[i, 0])].transpose(1, 2, 0), train_np[int(idx[i, 1])].transpose(1, 2, 0)), axis=1)
        canvas[i] = cat
    canvas = canvas.reshape(row, col, args.image_size, 2*args.image_size, args.channels)
    canvas = np.transpose(canvas, axes=(0, 2, 1, 3, 4))
    canvas = canvas.reshape(row*args.image_size, col*2*args.image_size, args.channels)

    # plt.figure(figsize=(30, 30))
    plt.imshow(canvas)
    plt.axis('off')
    plt.savefig(args.save_fig_path)
    plt.show()


if __name__ == '__main__':
    main()