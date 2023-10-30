from epsilon_star import epsilon_star
from utils import *
import matplotlib.pyplot as plt

import torch.nn.functional as F

# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" +
    (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

def compare_norm(args=args, p=2):
    gen_np = np.load(args.gen_path)
    loader = np.load(args.train_path)

    # model
    model, _, _ = load_model(args=args)
    model.eval()

    # x
    gen = torch.from_numpy(gen_np).to(device)

    # noise
    noise = torch.randn_like(gen).to(device)

    # time
    times = [i for i in range(1, 1001, 50)]
    norms = []
    norms_star = []
    for time in times:
        t = torch.full((args.batch_size,), time, device=device).long()
        x = q_sample(gen, t, noise).to(torch.float32)

        # here x is a batch of data! e.g. x.shape=[128, 3, 32, 32] --> norm.shape = [128,] --> norm.mean()
        norm = model(x, t, x_self_cond=None).reshape(args.batch_size, args.channels*args.image_size ** 2).norm(p=p, dim=1)
        norm_star = epsilon_star(x, t, loader, device).reshape(args.batch_size, args.channels*args.image_size ** 2).norm(p=p, dim=1)
    
        # diff = model(x, t, x_self_cond=None) - epsilon_star(x, t, loader, device)
        # norm = torch.norm(diff, p=p)
        norms.append(norm.mean().item())
        norms_star.append(norm_star.mean().item())
    
    plt.plot(times, norms, label='epsilon')
    plt.plot(times, norms_star, label='epsilon*')
    plt.legend()
    
    plt.xlabel('time')
    plt.ylabel('2-norm')
    plt.title('2-norm - time')
    plt.savefig(args.save_fig_path)
    plt.show()

def compare_cos_sim(args=args):
    gen_np = np.load(args.gen_path)
    loader = np.load(args.train_path)

    # model
    model, _, _ = load_model(args=args)
    model.eval()

    # x
    gen = torch.from_numpy(gen_np).to(device)

    # noise
    noise = torch.randn_like(gen).to(device)

    # time
    times = [i for i in range(1, 1001, 50)]

    # adapt size
    b = args.batch_size
    l = args.channels * args.image_size ** 2

    cos_sims = []
    for time in times:
        t = torch.full((args.batch_size,), time, device=device).long()
        x = q_sample(gen, t, noise).to(torch.float32)
        eps = model(x, t, x_self_cond=None)
        eps_star = epsilon_star(x, t, loader, device)
        cos_sim = F.cosine_similarity(eps.view(b, l), eps_star.view(b, l))
        cos_sims.append(cos_sim.mean().item())
    
    plt.plot(times, cos_sims)
    plt.xlabel('time')
    plt.ylabel('cosine similarity')

    plt.savefig(args.save_fig_path)
    plt.show()
    


def main():
    # compare_norm(args, 2)
    compare_cos_sim(args)

if __name__ == '__main__':
    main()
