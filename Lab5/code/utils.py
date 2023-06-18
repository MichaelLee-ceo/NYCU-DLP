import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

def kl_divergence(mu1, logvar1, mu2, logvar2, args):
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    KLD = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return KLD.sum() / args.batch_size

def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    # print('\ngt.shape:', gt.shape)
    # print('pred.shape:', pred.shape)

    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def pred(seq, cond, modules, args, device, posterior=False, prior=False):
    # reset h, c in lstm cell
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()

    if prior:
        modules['prior'].hidden = modules['prior'].init_hidden()

    seq = seq.permute(1, 0, 2, 3, 4)
    cond = cond.permute(1, 0, 2)

    pred_seq = [seq[0]]
    with torch.no_grad():
        x_t = seq[0]
        for i in range(1, args.n_past + args.n_future):
            if args.last_frame_skip or i <= args.n_past:
                h_t, skip = modules['encoder'](x_t)
            else:
                h_t, _ = modules['encoder'](x_t)

            cond_t = cond[i-1]

            if posterior:
                z_t, mu, sigma = modules['posterior'](h_t)
            elif prior:
                z_t, mu, sigma = modules['prior'](h_t)
            else:
                z_t = torch.randn((args.batch_size, args.z_dim)).to(device)

            info = torch.cat((cond_t, h_t, z_t), axis=1)
            g_t = modules['frame_predictor'](info)
            prediction = modules['decoder']([g_t, skip])
            
            if i < args.n_past:
                pred_seq.append(seq[i])
                x_t = seq[i]
            else:
                pred_seq.append(prediction)
                x_t = prediction
    
    pred_seq = torch.stack(pred_seq)
    # print('pred_seq.shape', pred_seq.shape)
    
    return pred_seq.permute(1, 0, 2, 3, 4)

def plot_pred(pred_seq, epoch, args, directory='.', name=''):
    save_image(pred_seq, './{}/gen/{}/epoch_{}.png'.format(args.log_dir, directory, epoch), nrow=12)
    

def plot_record(x, cond, modules, epoch, args, device, directory='.', prior=False):
    nsample = 5
    pred_list, psrn_list = [], []
    pred_list.append(x[0])

    for i in range(nsample):
        if i == 0:
            pred_seq = pred(x, cond, modules, args, device, posterior=True)
        else:
            pred_seq = pred(x, cond, modules, args, device, prior=prior)
        pred_list.append(pred_seq[0])

    ###### psnr ######
    gifs = [ [] for t in range(args.n_past + args.n_eval) ]
    text = [ [] for t in range(args.n_past + args.n_eval) ]

    for t in range(args.n_past + args.n_eval):
        # gt 
        gifs[t].append(add_border(pred_list[0][t], 'green'))
        text[t].append('Ground\ntruth')
        #posterior 
        if t < args.n_past:
            color = 'green'
        else:
            color = 'red'
        gifs[t].append(add_border(pred_list[1][t], color))
        text[t].append('Approx.\nposterior')
        # best 
        if t < args.n_past:
            color = 'green'
        else:
            color = 'red'
        gifs[t].append(add_border(pred_list[2][t], color))
        text[t].append('Best PSNR')
        # random 3
        for s in range(3):
            gifs[t].append(add_border(pred_list[s+3][t], color))
            text[t].append('Random\nsample %d' % (s+1))

    fname = '%s/gen/%s/result_%d.gif' % (args.log_dir, directory, epoch) 
    save_gif_with_text(fname, gifs, text)
    print("-- save prediction gif")

def plot_result(psnr_train_list, psnr_val_list, mse_list, kld_list, tfr_list, kl_list, args):
    epoch_full = np.arange(0, len(mse_list))
    epoch_sub = np.arange(0, len(mse_list), 5)
    epoch_sub = np.append(epoch_sub, epoch_full[-1])
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    plt.title('Training loss / Ratio curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss / psnr')
    ax1.set_ylim([0.0, 30.0])
    ax1.plot(epoch_full, mse_list, label='mse')
    ax1.plot(epoch_full, kld_list, label='kld')
    # ax1.plot(epoch_sub, psnr_train_list, c='limegreen', marker='o', label='psnr_train')
    ax1.plot(epoch_sub, psnr_val_list, c='gold', marker='o', label='psnr_val')
    ax1.legend(loc='center right')

    ax2.set_ylabel('ratio')
    ax2.plot(epoch_full, tfr_list, 'm--', label='Teacher forcing')
    ax2.plot(epoch_full, kl_list, '--', c='silver', label='KL weight')
    ax2.legend(loc='lower right')

    fig.tight_layout()
    plt.savefig('./{}/trainingCurve.png'.format(args.log_dir))
    print("-- save training figure")



def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))

def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

# def save_gif(filename, inputs, duration=0.25):
#     images = []
#     for tensor in inputs:
#         img = image_tensor(tensor, padding=0)
#         img = img.cpu()
#         img = img.transpose(0,1).transpose(1,2).clamp(0,1)
#         images.append(img.numpy())
#     imageio.mimsave(filename, images, duration=duration)

def save_gif_with_text(filename, inputs, text, duration=100):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1).numpy()
        images.append(np.uint8(img*255))
    imageio.mimsave(filename, images, duration=duration)

def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)

def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))

    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px