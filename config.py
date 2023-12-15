import tyro

import dataclasses

@dataclasses.dataclass
class Args:
    """Description.
    This should show up in the helptext!"""

    """string"""
    train_path: str = 'cifar_5k/cifar_subset_5k.npy'  # training set path   'xxx.npy'  e.g. [5000, 3, 32, 32]
    gen_path: str = 'results/cifar_5k/new_gen.npy'  # a batch of generated new images   'xxx.npy'  e.g. [256, 3, 32, 32]
    real_path: str = 'results/cifar_5k/train_nn.npy'  # a batch of training images which are respectively closest to generated images  'xxx.npy'  e.g. [256, 3, 32, 32]
    test_path: str = ''  # a batch of test images   'xx.npy'  e.g. [256, 3, 32, 32]
    save_np_path: str = 'gen_img.npy'  # save as '.npy'
    save_fig_path: str = 'fig.png'  # save as '.png'
    log_path: str = './epsilon_star/logs/celeba_10k'
    mode: str = 'trained'  # choose epsilon or epsilon*  --  trained  or  empirical
    choice: str = 'intp'  # only used in gd_x0.py, values in ['noise', 'gen', 'intp', 'test'] --> to choose the initialization of gradient descent

    results_folder: str = './results_cifar_5k'  # from where to load diffusion checkpoint
    folder: str = 'cifar_5k/cifar_5k_png'  # training data folder (in which images are all .png instead of .npy or other forms)

    gen_all_path: str = ''  # a batch of generated images, not necessarily different to train set!

    # logger
    log_name: str = './logs/cifar-10_subset/ddpm_gen_memorize.log'

    """int number"""
    # basic config
    SEED: int = 42
    ckpt: int = 50  # diffusion checkpoint
    time: int = 10  # time for epsilon(x, t)
    dim: int = 64  # size of unet
    
    # image config
    image_size: int = 32  # if square, h = w = image_size
    channels: int = 3  # gray -- 1, RGB -- 3
    batch_size: int = 256  # !!!!! not the trainloader batch_size !!!!!  --> actually is the batch_size of generated images
    train_batch_size: int = 16  # this one is trainloader batch_size

    # gradient descent config
    gd_epochs: int = 1000  # epochs of gradient descent

    # training and sampling
    train_num_steps: int = 50000  # training steps  !!not epochs!!
    sample_num: int = 20
    ddim_timesteps: int = 250  # 250 steps ddim sampling --> if ddpm, set ddim_timesteps = 1000 or any number larger than 1000
    num_fid_samples: int = 50000  # sample numbers for fid calculating

    # mix sample
    t_start: int = 100
    t_end: int = 200
    
    """float number"""
    train_lr: float = 1e-4  # train learning rate
    gd_lr: float = 1e-1  # gradient descent learning rate

    """bool"""
    need_sample: bool = False
    calculate_fid: bool = False


if __name__ == "__main__":
    args = tyro.cli(Args)
    print(args)