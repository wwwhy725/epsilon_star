from random import random
import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import numpy as np
import wandb

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F

from torchvision.utils import make_grid, save_image

from einops import rearrange, reduce, repeat

from denoising_diffusion_pytorch.version import __version__

from utils import *

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


class DiffusionStar(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        train_path,
        timesteps = 1000,
        train_batch_size = 16,
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

        self.self_condition = False

        # trained epsilon
        self.model = model
        self.channels = self.model.channels
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
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

    # empirical min
    """considering that the dataloader may be very large, we use batch to avoid OOM"""
    def epsilon_star_batch(self, x, t, dataloader, batch_size=2500):
        device = self.device
        b, c, h, w = x.shape  # take [256, 3, 32, 32] as an example
        num_train = dataloader.shape[0]  # take 5k as an example
        num_batches = (num_train + batch_size - 1) // batch_size

        coef_deno = 1. / (-2 * (1. - self.alphas_cumprod))
        coef_x = 1. / torch.sqrt(1. - self.alphas_cumprod)
        coef_x_hat = self.sqrt_alphas_cumprod / torch.sqrt(1. - self.alphas_cumprod)
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
            x_minus_x0 = x[:, None, :, :, :] - extract(sqrt_alphas_cumprod, t, batch_train_data.shape) * batch_train_data  # [256, batch_size, 3, 32, 32]
            norm = x_minus_x0.norm(p=2, dim=(2, 3, 4)) ** 2   # [256, batch_size]
            norm = extract(coef_deno, t, norm.shape) * norm
            # max of norm
            max_norm = torch.max(max_norm, norm.max(dim=1)[0])

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_train)
            train_data = torch.from_numpy(dataloader[start_idx:end_idx]).to(device)  # a batch of train data  [batch_size, 3, 32, 32]
            batch_train_data = train_data.repeat(b, 1, 1, 1).reshape(b, train_data.shape[0], c, h, w)  # [256, batch_size, 3, 32, 32]
            x_minus_x0 = x[:, None, :, :, :] - extract(sqrt_alphas_cumprod, t, batch_train_data.shape) * batch_train_data  # [256, batch_size, 3, 32, 32]
            norm = x_minus_x0.norm(p=2, dim=(2, 3, 4)) ** 2   # [256, batch_size]
            norm = extract(coef_deno, t, norm.shape) * norm
            # softmax
            # numerator -- a tensor: \sum [exp()*xi]
            numerator += (torch.exp(norm - max_norm[:, None])[:, :, None, None, None] * train_data[None, :, :, :, :]).sum(dim=1)  # [256, 3, 32, 32]

            # denominator -- a real number \sum exp()
            denominator += torch.sum(torch.exp(norm - max_norm[:, None]), dim=1)  # [256, ]
    
        weighted_x = numerator / denominator[:, None, None, None]    
        result = extract(coef_x, t, x.shape) * x - extract(coef_x_hat, t, weighted_x.shape) * weighted_x

        return result

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

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
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
    def sample(self, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop
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

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        eps_star = self.epsilon_star_batch(x, t, self.loader)

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = eps_star
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = F.smooth_l1_loss(model_out, target, reduction='none')  # tbd smooth L1 loss!!
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, 1000, (b,), device=device).long() 
        # img = self.normalize(img)
        img = img.to(torch.float32)

        return self.p_losses(img, t, *args, **kwargs)


def load_model_star(args):
    if args.channels == 1:
        model = Unet(
        dim = 64,
        dim_mults = (1, 2),
        channels = args.channels,
        full_attn = (False, True),
        flash_attn = True
        )
    elif args.channels == 3:
        model = Unet(
        dim = args.dim,
        dim_mults = (1, 2, 2, 2),  # change here according to your need
        channels = args.channels,
        flash_attn = True
        )
    else:
        raise ValueError('image channels have to be either 1 or 3!')

    diffusion = DiffusionStar(
        model=model,
        image_size=args.image_size,
        train_path=args.train_path
    )

    trainer = Trainer(
        diffusion,
        args.folder,
        train_batch_size = args.train_batch_size,
        train_lr = args.train_lr,
        train_num_steps = args.train_num_steps,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        num_samples = 256,
        results_folder = args.results_folder,
        amp = False,                       # turn on mixed precision
        calculate_fid = args.calculate_fid,  # whether to calculate fid during training
        num_fid_samples=args.num_fid_samples
    )
    trainer.load(args.ckpt)
    
    return model, diffusion, trainer

def train_pred_eps_star():
    """  use Dataset_fromnp to train!!  """
    model = Unet(
        dim = 256,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        flash_attn = True
    )
    diffusion = DiffusionStar(
        model,
        image_size = 32,
        timesteps = 1000,           # number of steps
        train_path = 'epsilon_star/cifar_2cls_10k/cifar_10k.npy',#args.train_path,
        objective = 'pred_noise',
        beta_schedule = 'linear'
    )
    trainer = Trainer(
        diffusion,
        folder='epsilon_star/cifar_2cls_10k/cifar_10k.npy',  
        train_batch_size = 128,    
        train_lr = 1e-4,
        train_num_steps = 120000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.9999,                # exponential moving average decay
        num_samples = 16,
        results_folder = '/mnt/store/lyx/github_projs/why/DDPM/results_cifar_10k_cls2_star_half/debug',  
        amp = False,                       # turn on mixed precision
        num_fid_samples = 10000,
        calculate_fid = False,              # whether to calculate fid during training
        save_best_and_latest_only = False
    )
    # trainer.load(80)
    # print(trainer.step)
    trainer.train()

def sample():
    _, diffusion, _ = load_model_star(args)
    gen = diffusion.sample(batch_size=256)
    grid = make_grid(gen, nrow = 16)
    save_image(grid, args.save_fig_path)

if __name__ == '__main__':
    train_pred_eps_star()
    # sample()
    