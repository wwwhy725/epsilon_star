import tyro

import dataclasses

@dataclasses.dataclass
class Args:
    """Description.
    This should show up in the helptext!"""

    """string"""
    train_path: str = ''  # training set path   'xxx.npy'  e.g. [5000, 3, 32, 32]
    gen_path: str = ''  # a batch of generated new images   'xxx.npy'  e.g. [256, 3, 32, 32]
    real_path: str = ''  # a batch of training images which are respectively closest to generated images  'xxx.npy'  e.g. [256, 3, 32, 32]
    test_path: str = ''  # a batch of test images   'xx.npy'  e.g. [256, 3, 32, 32]
    save_np_path: str = ''  # save as '.npy'
    save_fig_path: str = 'result_gd/far_t10_iter300_train_interpolation_2.png'  # save as '.png'
    mode: str = 'trained'  # choose epsilon or epsilon*  --  trained  or  empirical
    choice: str = 'gen'  # only used in gd_x0.py, values in ['noise', 'gen', 'intp', 'test'] --> to choose the initialization of gradient descent

    results_folder: str = './results_mnist50_1wepoch_1e-4lr'  # from where to load diffusion checkpoint
    folder: str = 'mnist_50'  # training data folder (in which images are all .png instead of .npy or other forms)

    gen_all_path: str = ''  # a batch of generated images, not necessarily different to train set!

    # logger
    log_name: str = './logs/cifar-10_subset/ddpm_gen_memorize.log'

    """int number"""
    # basic config
    SEED: int = 42
    ckpt: int = 10  # diffusion checkpoint
    time: int = 10  # time for epsilon(x, t)
    
    # image config
    image_size: int = 32  # if square, h = w = image_size
    channels: int = 3  # gray -- 1, RGB -- 3
    batch_size: int = 256  # !!!!! not the trainloader batch_size !!!!!  --> actually is the batch_size of generated images
    train_batch_size: int = 1024  # this one is trainloader batch_size

    # gradient descent config
    gd_epochs: int = 1000  # epochs of gradient descent

    # training and sampling
    train_num_steps: int = 50000  # training steps  !!not epochs!!
    sample_num: int = 20
    t_start: int = 100  # mix epsilon and epsilon* to sample -- from when to start epsilon*
    t_end: int = 200  # mix epsilon and epsilon* -- from when to end epsilon*
    
    """float number"""
    train_lr: float = 1e-4  # train learning rate
    gd_lr: float = 1e-1  # gradient descent learning rate

    """tuple"""
    dim_mults: tuple = (1, 2, 4, 8)  # in UNet   [if channels = 1, dim_mults = (1, 2)]
    full_attn: tuple = (False, False, False, True)  # in UNet  [if channels = 1, full_attn = (False, True)]


if __name__ == "__main__":
    args = tyro.cli(Args)
    print(args)