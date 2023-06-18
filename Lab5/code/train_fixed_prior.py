import argparse
import math
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import *

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=24, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=20, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=4, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=10, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--cond_dim', type=int, default=7, help='dimensionality of condition input vector')
    parser.add_argument('--beta', type=float, default=1, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')  

    args = parser.parse_args()
    return args

def train(x, cond, modules, optimizer, kl_anneal, args):
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()
    mse_loss = nn.MSELoss()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0

    '''
        lstm default: batch_first=False => [seq_len, batch, input_size]
        x.shape: [batch:0, seq_len:1, channel:2, width:3, height:4]
              -> [seq_len:1, batch:0, channel:2, width:3, height:4]

        cond.shape: [batch:0, seq_len:1, action+position:2]
                 -> [seq_len:1, batch:0, action+position:2]
    '''

    x = x.permute(1, 0, 2, 3, 4)
    cond = cond.permute(1, 0, 2)
    # print('Seq:', x.shape)
    # print('cond:', cond.shape)
    
    use_teacher_forcing = True if random.random() < args.tfr else False
    x_t_1 = x[0]
    for i in range(1, args.n_past + args.n_future):
        x_t = x[i]
        h_t, _ = modules['encoder'](x_t)
        z_t, mu, sigma = modules['posterior'](h_t)
        
        if args.last_frame_skip or i <= args.n_past:
            h_t_1, skip = modules['encoder'](x_t_1)
        else:
            h_t_1, _ = modules['encoder'](x_t_1)

        cond_t_1 = cond[i-1]

        info = torch.cat((cond_t_1, h_t_1, z_t), axis=1)
        g_t = modules['frame_predictor'](info)
        prediction = modules['decoder']([g_t, skip])

        if use_teacher_forcing:
            x_t_1 = x[i]
        else:
            x_t_1 = prediction

        mse += mse_loss(x_t, prediction)
        kld += kl_criterion(mu, sigma, args)

    beta = kl_anneal.get_beta()
    loss = mse + kld * beta
    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        self.cyclical = args.kl_anneal_cyclical
        self.cycles = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.num_iters = args.niter
        self.idx = 0

        if self.cyclical == False:
            self.cycles = 1
            self.ratio = 0.25

        self.L = np.ones(self.num_iters) * args.beta
        period = self.num_iters / self.cycles
        step = args.beta / (period * self.ratio)

        for i in range(self.cycles):
            beta = 0
            for j in range(math.ceil(period * self.ratio)):
                if beta >= 1:
                    break
                self.L[int(j + i*period)] = beta
                beta += step

    def update(self):
        self.idx += 1
    
    def get_beta(self):
        curr_beta = self.L[self.idx]
        return curr_beta


def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        print("===== Load model from checkpoint: {}/model.pth =====".format(args.model_dir))
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        # args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        print("===== Start with empty model =====")
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f-tfr_start=%d-cyclical=%s'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta, args.tfr_start_decay_epoch, args.kl_anneal_cyclical)

        # name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f-notfr-constkl'\
            # % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta)


        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 刪掉舊的 train_record.txt，重新記錄 log
    if args.model_dir == '':
        if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
            os.remove('./{}/train_record.txt'.format(args.log_dir))
        if os.path.exists('./{}/record.txt'.format(args.log_dir)):
            os.remove('./{}/record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    with open('./{}/record.txt'.format(args.log_dir), 'a') as record:
        record.write('args: {}\n'.format(args))

    # ------------ build the models --------------
    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    else:
        frame_predictor = lstm(args.g_dim+args.z_dim+args.cond_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    validate_iterator = iter(validate_loader)
    print('train_data samples: {}, validation_data samples: {}'.format(len(train_data), len(validate_data)))

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }

    # --------- training loop ------------------------------------
    # args.epoch_size = len(train_loader)
    best_val_psnr = 0
    slope = (args.tfr_lower_bound - args.tfr) / (args.niter - args.tfr_start_decay_epoch)

    train_mse, train_kld = [], []
    train_psnr_list, val_psnr_list = [], []
    tfr_list, kl_list = [], []

    progress = tqdm(total=args.niter)
    for epoch in range(start_epoch, start_epoch + niter):
        frame_predictor.train()
        posterior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0
        train_psnr = []

        # print("> KL beta: {:.2f}, Teacher forcing ratio: {:.2f}".format(kl_anneal.get_beta(), args.tfr))
        for _ in tqdm(range(args.epoch_size), leave=False):
            try: 
                seq, cond = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)

            seq, cond = seq.to(device), cond.to(device)
            loss, mse, kld = train(seq, cond, modules, optimizer, kl_anneal, args)

            # if epoch % 5 == 0 or epoch == (start_epoch + niter - 1):
            #     train_pred_seq = pred(seq, cond, modules, args, device)
            #     _, _, psnr = finn_eval_seq(seq.permute(1, 0, 2, 3, 4)[args.n_past:], train_pred_seq.permute(1, 0, 2, 3, 4)[args.n_past:])
            #     train_psnr.append(psnr)

            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld

        # if epoch % 5 == 0 or epoch == (start_epoch + niter - 1):
        #     train_psnr_list.append(np.mean(np.concatenate(train_psnr)))

        train_mse.append(epoch_mse / args.epoch_size)
        train_kld.append(epoch_kld / args.epoch_size)
        tfr_list.append(args.tfr)
        kl_list.append(kl_anneal.get_beta())

        kl_anneal.update()
        if epoch >= args.tfr_start_decay_epoch:         ### Update teacher forcing ratio ###
            tfr = slope * (epoch - args.tfr_start_decay_epoch) + 1
            args.tfr = max(tfr, args.tfr_lower_bound)

        progress.update(1)
        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n' % (epoch, epoch_loss  / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size)))
        
        with open('./{}/record.txt'.format(args.log_dir), 'a') as record:
            record.write(('[epoch: %02d] mse loss: %.5f | kld loss: %.5f | tfr: %.2f | klw: %.2f\n' % (epoch, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size, tfr_list[-1], kl_list[-1])))

        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

        # validation
        if epoch % 5 == 0 or epoch == (start_epoch + niter - 1):
            val_psnr = []
            for i, (validate_seq, validate_cond) in enumerate(validate_loader):
                validate_seq, validate_cond = validate_seq.to(device), validate_cond.to(device)
                pred_seq = pred(validate_seq, validate_cond, modules, args, device)

                if i == 5:
                    plot_pred(pred_seq[0], epoch, args)                                             # save to png
                    plot_record(validate_seq, validate_cond, modules, epoch, args, device)    # save to gif

                validate_seq, pred_seq = validate_seq.permute(1, 0, 2, 3, 4), pred_seq.permute(1, 0, 2, 3, 4)
                _, _, psnr = finn_eval_seq(validate_seq[args.n_past:], pred_seq[args.n_past:])
                val_psnr.append(psnr)

            val_psnr_list.append(np.mean(np.concatenate(val_psnr)))

            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                # train_record.write(('====================== train psnr = {:.5f} ========================\n'.format(train_psnr_list[-1])))
                train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(val_psnr_list[-1])))

            with open('./{}/record.txt'.format(args.log_dir), 'a') as record:
                record.write(('====================== validate psnr = {:.5f} ========================\n'.format(val_psnr_list[-1])))
            
            if val_psnr_list[-1] > best_val_psnr:
                best_val_psnr = val_psnr_list[-1]
                # save the model
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'args': args,
                    'last_epoch': epoch},
                    '%s/model.pth' % args.log_dir)
        '''
        if epoch % 20 == 0 or epoch == (start_epoch + niter - 1):
            try:
                validate_seq, validate_cond = next(validate_iterator)
            except StopIteration:
                validate_iterator = iter(validate_loader)
                validate_seq, validate_cond = next(validate_iterator)

                plot_record(validate_seq, validate_cond, modules, epoch, args)
        '''

    plot_result(train_psnr_list, val_psnr_list, train_mse, train_kld, tfr_list, kl_list, args)

if __name__ == '__main__':
    main()