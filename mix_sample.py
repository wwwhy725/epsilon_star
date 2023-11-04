import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import math
import copy
from pathlib import Path
from random import random
from utils import *
from epsilon_star_sample import *
import sample_memo

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from denoising_diffusion_pytorch.version import __version__


class DiffusionStar(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels,
        train_path,
        timesteps = 1000,
        train_batch_size = 16,
        t_start = 100,
        t_end = 200,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'linear',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.channels = channels
        self.self_condition = False

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs).to(self.device)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        # train loader
        self.train_batch_size = train_batch_size
        # self.loader = get_trainloader(self.image_size, self.channels, batch_size=self.train_batch_size, train_path=train_path)
        self.loader = np.load(train_path)

        # mix sample
        self.t_start = t_start
        self.t_end = t_end

        # trained epsilon
        self.model = model

    # empirical min
    """considering that the dataloader may be very large, we use batch to avoid OOM"""
    def epsilon_star_batch(self, x, t, dataloader, batch_size=5000):
        device = self.device
        b, c, h, w = x.shape  # take [256, 3, 32, 32] as an example
        num_train = dataloader.shape[0]  # take 5k as an example
        num_batches = (num_train + batch_size - 1) // batch_size

        time = t[0].item()  # int number
        alpha_bar = self.alphas_cumprod[time]  # float number
        # *********************************
        numerator = torch.zeros(b, c, h, w).to(device)  # [256, 3, 32, 32]
        denominator = torch.zeros(b).to(device)
        # deal with batch
        max_norm = -torch.inf * torch.ones(b).to(device)
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_train)
            train_data = torch.from_numpy(dataloader[start_idx:end_idx]).to(device)  # a batch of train data  [batch_size, 3, 32, 32]
            batch_train_data = train_data.repeat(b, 1, 1, 1).reshape(b, train_data.shape[0], c, h, w)  # [256, batch_size, 3, 32, 32]
            x_minus_x0 = x[:, None, :, :, :] - torch.sqrt(alpha_bar) * batch_train_data  # [256, batch_size, 3, 32, 32]
            norm = x_minus_x0.norm(p=2, dim=(2, 3, 4)) ** 2 / (-2 * (1 - alpha_bar))  # [256, batch_size]

            # max of norm
            max_norm = torch.max(max_norm, norm.max(dim=1)[0])

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_train)
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
    
    # empirical min
    """if the dataloader is not that large, use this one to accelerate sampling"""
    def epsilon_star(self, x, t, dataloader):
        device = self.device
        b, c, h, w = x.shape  # take [256, 3, 32, 32] as an example
        # *********************************
        time = t[0].item()
        alpha_bar = self.alphas_cumprod[time]
        train_data = torch.from_numpy(dataloader).to(device)
        num_train = train_data.shape[0]  # take 5k as an example
        batch_train_data = train_data.repeat(b, 1, 1, 1).reshape(b, num_train, c, h, w)  # [256, 5k, 3, 32, 32]

        x_minus_x0 = x[:, None, :, :, :] - torch.sqrt(alpha_bar) * batch_train_data  # [256, 5k, 3, 32, 32]
        norm = x_minus_x0.norm(p=2, dim=(2, 3, 4)) ** 2 / (-2 * (1 - alpha_bar))  # [256, 5k]

        softmax = F.softmax(norm, dim=1)
        weighted_x = (softmax[:, :, None, None, None] * train_data[None, :, :, :, :]).sum(dim=1)  # [256, 3, 32, 32]

        return x / torch.sqrt(1 - alpha_bar) - torch.sqrt(alpha_bar) / torch.sqrt(1 - alpha_bar) * weighted_x

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):  # tbd  !!!  here the mix sampling
        time = int(t[0].item())
        if self.t_start < time and time < self.t_end:
            model_output = self.epsilon_star_batch(x, t, self.loader)  # here use batch if OOM
        else:
            model_output = self.model(x, t, x_self_cond)
        # model_output = self.epsilon_star(x, t, self.loader)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps, mininterval=10):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps)

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.epsilon_star(x, t, self.loader)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)


def mix_sample(args=args):
    model, _, _ = load_model(args=args)
    mix_diffusion = DiffusionStar(
        model,
        image_size=args.image_size,
        channels=args.channels,
        train_path=args.train_path,
        t_start=args.t_start,
        t_end=args.t_end
        # train_batch_size=train_batch_size,
        # sampling_timesteps=200  # 200 timesteps ddim
    )
    gen = mix_diffusion.sample(batch_size=args.batch_size)

    gen_np = gen.cpu().detach().numpy()
    np.save(args.save_np_path, gen_np)

    return gen

if __name__ == '__main__':
    # mix sample
    gen = mix_sample(args)

    # calculate memorization and visualize
    sample_memo.sample_memo_visualization(gen, args=args)