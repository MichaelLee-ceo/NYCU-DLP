import os
import random
import argparse
import numpy as np

import torch
from diffusers import DDPMScheduler
from model import MyConditionedUNet
from utils import make_gif

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default="./data", type=str)
parser.add_argument('--log_dir', default="logs", type=str)
parser.add_argument('--gif_dir', default="gifs", type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--sample_size', default=64, type=int)
parser.add_argument('--beta_schedule', default="linear", type=str)
parser.add_argument('--predict_type', default="epsilon", type=str)
parser.add_argument('--block_dim', default=128, type=int)
parser.add_argument('--layers_per_block', default=2, type=int)
parser.add_argument('--embed_type', default="timestep", type=str)
parser.add_argument('--lr_warmup_steps', default=500, type=int)
parser.add_argument('--num_inference_steps', default=50, type=int)
parser.add_argument('--seed', default=1, type=int)
args = parser.parse_args()

# set reproducibility
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("> Using device: {}".format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

name = 'lr={:.5f}-lr_warmup={}-block_dim={}-layers={}-embed={}-schedule={}-predict_type={}'.format(args.lr, args.lr_warmup_steps, args.block_dim, args.layers_per_block, args.embed_type, args.beta_schedule, args.predict_type)
args.log_dir = './%s/%s' % (args.log_dir, name)
args.gif_dir = '%s/%s' % (args.log_dir, args.gif_dir)

os.makedirs(args.gif_dir, exist_ok=True)

sample_size = args.sample_size
block_dim = args.block_dim
layers = args.layers_per_block

model = MyConditionedUNet(
    sample_size=sample_size,       # the target image resolution
    in_channels=3,                 # additional input channels for class condition
    out_channels=3,
    layers_per_block=layers,
    block_out_channels=(block_dim, block_dim, block_dim*2, block_dim*2, block_dim*4, block_dim*4),
    down_block_types=(
        "DownBlock2D",          # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",      # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",        # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",            # a regular ResNet upsampling block
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(device)

checkpoint = torch.load(os.path.join(args.log_dir, "model.pth"))
model.load_state_dict(checkpoint["model"])

beta_schedule = "squaredcos_cap_v2" if args.beta_schedule == "cosine" else "linear"
scheduler = DDPMScheduler(beta_schedule=beta_schedule)
scheduler.set_timesteps(args.num_inference_steps)

make_gif(model, scheduler, args, device, "test")
make_gif(model, scheduler, args, device, "new_test")