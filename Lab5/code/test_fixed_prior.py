import argparse
import random
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import *

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--batch_size', default=24, type=int, help='batch size')
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--n_eval', type=int, default=10, help='number of frames to predict at eval time')
parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
args = parser.parse_args()

# args.model_dir = 'rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=1.0000000-notfr-constkl'

args.model_dir = 'rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=1.0000000-tfr_start=20-cyclical=False'
args.log_dir = '%s/%s' % (args.log_dir, args.model_dir)
saved_model = torch.load('%s/model.pth' % args.log_dir)
print('loading from:', args.model_dir, 'best_epoch:', saved_model['last_epoch'])

os.makedirs('%s/gen/test/' % args.log_dir, exist_ok=True)

frame_predictor = saved_model['frame_predictor'].to(device)
posterior = saved_model['posterior'].to(device)
encoder = saved_model['encoder'].to(device)
decoder = saved_model['decoder'].to(device)

modules = {
    'frame_predictor': frame_predictor,
    'posterior': posterior,
    'encoder': encoder,
    'decoder': decoder,
}

frame_predictor.eval()
encoder.eval()
decoder.eval()
posterior.eval()


test_data = bair_robot_pushing_dataset(args, 'test')
test_loader = DataLoader(test_data,
                        # num_workers=4,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True,
                        pin_memory=True)

test_psnr = []
with torch.no_grad():
    for i, (seq, cond) in enumerate(test_loader):
        seq, cond = seq.to(device), cond.to(device)
        pred_seq = pred(seq, cond, modules, args, device)

        # save ground truth and pred_seq
        save_image(torch.cat((seq[0], pred_seq[0])), './{}/gen/test/test_{}.png'.format(args.log_dir, i), nrow=12)
        plot_record(seq, cond, modules, i, args, device, directory='test')

        seq, pred_seq = seq.permute(1, 0, 2, 3, 4), pred_seq.permute(1, 0, 2, 3, 4)
        _, _, psnr = finn_eval_seq(seq[args.n_past:], pred_seq[args.n_past:])
        test_psnr.append(psnr)
    
avg_psnr = np.mean(test_psnr)
print("Test PSNR:", avg_psnr)