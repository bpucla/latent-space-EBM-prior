import sys
import os

def set_gpu(gpu):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import argparse
import json
import random
import shutil
import copy
import logging
import datetime
import pickle
import itertools
import time
import math

import numpy as np

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn

import torchvision
import torchvision.transforms as transforms

import pygrid


##########################################################################################################
## Parameters

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')
    parser.add_argument('--gpu_multi', type=bool, default=True, help='mutli gpu')

    parser.add_argument('--dataset', type=str, default='celeba128', choices=['svhn', 'celeba', 'celeba_crop', 'celeba32_sri', 'celeba64_sri', 'celeba64_sri_crop', 'celeba128'])
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--batch_size', default=int(4*100), type=int)

    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nc', default=3)

    parser.add_argument('--nez', default=1, help='size of the output of ebm')
    parser.add_argument('--ngf', default=128, help='feature dimensions of generator')
    parser.add_argument('--ndf', default=1000, help='feature dimensions of ebm')

    parser.add_argument('--e_prior_sig', type=float, default=1, help='prior of ebm z')
    parser.add_argument('--e_init_sig', type=float, default=1, help='sigma of initial distribution')
    parser.add_argument('--e_activation', type=str, default='gelu', choices=['gelu', 'lrelu', 'swish', 'mish'])
    parser.add_argument('--e_activation_leak', type=float, default=0.2)
    parser.add_argument('--e_energy_form', default='identity', choices=['identity', 'tanh', 'sigmoid', 'softplus'])
    parser.add_argument('--e_l_steps', type=int, default=80, help='number of langevin steps')
    parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--e_sn', default=False, type=bool, help='spectral regularization')

    parser.add_argument('--g_llhd_sigma', type=float, default=0.3, help='prior of factor analysis')
    parser.add_argument('--g_activation', type=str, default='lrelu')
    parser.add_argument('--g_l_steps', type=int, default=40, help='number of langevin steps')
    parser.add_argument('--g_l_step_size', type=float, default=0.1, help='stepsize of langevin')
    parser.add_argument('--g_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--g_batchnorm', default=False, type=bool, help='batch norm')

    parser.add_argument('--e_lr', default=0.00002, type=float)
    parser.add_argument('--g_lr', default=0.0001, type=float)

    parser.add_argument('--e_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--g_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')

    parser.add_argument('--e_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--g_max_norm', type=float, default=100, help='max norm allowed')

    parser.add_argument('--e_decay', default=0, help='weight decay for ebm')
    parser.add_argument('--g_decay',  default=0, help='weight decay for gen')

    parser.add_argument('--e_gamma', default=0.998, help='lr decay for ebm')
    parser.add_argument('--g_gamma', default=0.998, help='lr decay for gen')

    parser.add_argument('--g_beta1', default=0.5, type=float)
    parser.add_argument('--g_beta2', default=0.999, type=float)

    parser.add_argument('--e_beta1', default=0.5, type=float)
    parser.add_argument('--e_beta2', default=0.999, type=float)

    parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs to train for') # TODO(nijkamp): set to >100
    # parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--n_printout', type=int, default=25, help='printout each n iterations')
    parser.add_argument('--n_plot', type=int, default=1, help='plot each n epochs')
    parser.add_argument('--n_ckpt', type=int, default=20, help='save ckpt each n epochs')
    parser.add_argument('--n_metrics', type=int, default=399, help='fid each n epochs')
    # parser.add_argument('--n_metrics', type=int, default=1, help='fid each n epochs')
    parser.add_argument('--n_stats', type=int, default=1, help='stats each n epochs')

    parser.add_argument('--n_fid_samples', type=int, default=30000) # TODO(nijkamp): we used 40,000 in short-run inference
    # parser.add_argument('--n_fid_samples', type=int, default=1000)

    parser.add_argument('--load_ckpt', type=str, default=None)
    parser.add_argument('--eval', type=bool, default=False)

    return parser.parse_args()


def create_args_grid():
    # TODO add your enumeration of parameters here

    # e_lr = [0.00002, 0.00005]
    # e_l_step_size = [0.2, 0.4, 0.8]
    # e_init_sig = [2.0, 3.0, 1.0]
    # e_l_steps = [40, 60]

    # g_llhd_sigma = [0.2, 0.3]
    # g_lr = [0.0001, 0.00005]
    # g_l_steps = [20, 40]
    # g_activation = ['lrelu', 'gelu']

    e_lr = [0.00002]
    e_l_step_size = [0.4]
    e_init_sig = [1.0]
    # e_l_steps = [40, 60]
    e_l_steps = [60]
    e_activation = ['gelu']
    # e_activation = ['gelu', 'lrelu']

    g_llhd_sigma = [0.3]
    g_lr = [0.0001]
    g_l_steps = [20, 40]
    g_activation = ['lrelu']

    ngf = [64, 128]

    args_list = [e_lr, e_l_step_size, e_init_sig, e_l_steps, e_activation, g_llhd_sigma, g_lr, g_l_steps, g_activation, ngf]

    opt_list = []
    for i, args in enumerate(itertools.product(*args_list)):
        opt_job = {'job_id': int(i), 'status': 'open'}
        opt_args = {
            'e_lr': args[0],
            'e_l_step_size': args[1],
            'e_init_sig': args[2],
            'e_l_steps': args[3],
            'e_activation': args[4],
            'g_llhd_sigma': args[5],
            'g_lr': args[6],
            'g_l_steps': args[7],
            'g_activation': args[8],
            'ngf': args[9],
        }
        # TODO add your result metric here
        opt_result = {'fid_best': 0.0, 'fid': 0.0, 'mse': 0.0}

        opt_list += [merge_dicts(opt_job, opt_args, opt_result)]

    return opt_list


def update_job_result(job_opt, job_stats):
    # TODO add your result metric here
    job_opt['fid_best'] = job_stats['fid_best']
    job_opt['fid'] = job_stats['fid']
    job_opt['mse'] = job_stats['mse']


##########################################################################################################
## Data

def get_dataset(args):

    fs_prefix = './' if not is_xsede() else '/pylon5/ac5fpjp/bopang/ebm_prior/'

    if args.dataset == 'svhn':
        import torchvision.transforms as transforms
        ds_train = torchvision.datasets.SVHN(fs_prefix + 'data/{}'.format(args.dataset), download=True,
                                             transform=transforms.Compose([
                                             transforms.Resize(args.img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        ds_val = torchvision.datasets.SVHN(fs_prefix + 'data/{}'.format(args.dataset), download=True, split='test',
                                             transform=transforms.Compose([
                                             transforms.Resize(args.img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        return ds_train, ds_val

    if args.dataset == 'celeba':

        import torchvision.transforms as transforms

        ds_train = torchvision.datasets.CelebA(fs_prefix + 'data/{}/train'.format(args.dataset), split='train', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Resize(args.img_size),
                                                        transforms.CenterCrop(args.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        ds_val = torchvision.datasets.CelebA(fs_prefix + 'data/{}/val'.format(args.dataset), split='valid', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Resize(args.img_size),
                                                        transforms.CenterCrop(args.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return ds_train, ds_val

    if args.dataset == 'celeba_crop':

        crop = lambda x: transforms.functional.crop(x, 45, 25, 173-45, 153-25)

        import torchvision.transforms as transforms

        ds_train = torchvision.datasets.CelebA(fs_prefix + 'data/{}/train'.format(args.dataset), split='train', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Lambda(crop),
                                                        transforms.Resize(args.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        ds_val = torchvision.datasets.CelebA(fs_prefix + 'data/{}/val'.format(args.dataset), split='valid', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Lambda(crop),
                                                        transforms.Resize(args.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return ds_train, ds_val

    elif args.dataset == 'celeba32_sri':

        data_path = fs_prefix + 'data/{}/img_align_celeba'.format(args.dataset)
        cache_pkl = fs_prefix + 'data/{}/celeba_40000_32.pickle'.format(args.dataset)

        from data import SingleImagesFolderMTDataset
        import PIL
        import torchvision.transforms as transforms

        ds_train = SingleImagesFolderMTDataset(root=data_path,
                                            cache=cache_pkl,
                                            num_images=40000,
                                            transform=transforms.Compose([
                                                PIL.Image.fromarray,
                                                transforms.Resize(args.img_size),
                                                transforms.CenterCrop(args.img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ]))

        # TODO(nijkamp): create ds_val pickle
        ds_val = ds_train

        return ds_train, ds_val

    elif args.dataset == 'celeba64_sri':

        # wget https://www.dropbox.com/s/zjcpa1hrjxy9nne/celeba64_40000.pkl?dl=1

        data_path = fs_prefix + 'data/{}/img_align_celeba'.format(args.dataset)
        cache_pkl = fs_prefix + 'data/{}/celeba64_40000.pkl'.format(args.dataset)

        assert os.path.exists(cache_pkl)

        from data import SingleImagesFolderMTDataset
        import PIL
        import torchvision.transforms as transforms

        ds_train = SingleImagesFolderMTDataset(root=data_path,
                                            cache=cache_pkl,
                                            num_images=40000,
                                            transform=transforms.Compose([
                                                PIL.Image.fromarray,
                                                transforms.Resize(args.img_size),
                                                transforms.CenterCrop(args.img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ]))

        # TODO(nijkamp): create ds_val pickle
        ds_val = ds_train

        return ds_train, ds_val

    elif args.dataset == 'celeba64_sri_crop':

        # wget https://www.dropbox.com/s/9omncogiyaul54d/celeba_40000_64_center.pickle?dl=0

        data_path = fs_prefix + 'data/{}/img_align_celeba'.format(args.dataset)
        cache_pkl = fs_prefix + 'data/{}/celeba_40000_64_center.pickle'.format(args.dataset)

        from data import SingleImagesFolderMTDataset
        import PIL
        import torchvision.transforms as transforms

        ds_train = SingleImagesFolderMTDataset(root=data_path,
                                            cache=cache_pkl,
                                            num_images=40000,
                                            transform=transforms.Compose([
                                                PIL.Image.fromarray,
                                                transforms.Resize(args.img_size),
                                                transforms.CenterCrop(args.img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ]))

        # TODO(nijkamp): create ds_val pickle
        ds_val = ds_train

        return ds_train, ds_val

    elif args.dataset == 'celeba128':

        # wget https://www.dropbox.com/s/9omncogiyaul54d/celeba_40000_64_center.pickle?dl=0

        data_path = fs_prefix + 'data/{}/img_align_celeba'.format(args.dataset)
        cache_pkl = fs_prefix + 'data/{}/celeba_30000_128.pickle'.format(args.dataset)

        from data import SingleImagesFolderMTDataset
        import PIL
        import torchvision.transforms as transforms

        ds_train = SingleImagesFolderMTDataset(root=data_path,
                                            cache=cache_pkl,
                                            num_images=30000,
                                            transform=transforms.Compose([
                                                PIL.Image.fromarray,
                                                transforms.Resize(args.img_size),
                                                transforms.CenterCrop(args.img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ]))

        # TODO(nijkamp): create ds_val pickle
        ds_val = ds_train

        return ds_train, ds_val

    else:
        raise ValueError(args.dataset)

##########################################################################################################
## Model

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)

def get_activation(name, args):
    return {'gelu': GELU(), 'lrelu': nn.LeakyReLU(args.e_activation_leak), 'mish': Mish(), 'swish': Swish()}[name]


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, args):
        super().__init__()

        f = get_activation(args.g_activation, args)

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(args.nz, args.ngf*16, 4, 1, 0, bias = not args.g_batchnorm),
            nn.BatchNorm2d(args.ngf*4) if args.g_batchnorm else nn.Identity(),
            f,

            nn.ConvTranspose2d(args.ngf*16, args.ngf*8, 4, 2, 1, bias = not args.g_batchnorm),
            nn.BatchNorm2d(args.gnf*4) if args.g_batchnorm else nn.Identity(),
            f,

            nn.ConvTranspose2d(args.ngf*8, args.ngf*4, 4, 2, 1, bias = not args.g_batchnorm),
            nn.BatchNorm2d(args.gnf*4) if args.g_batchnorm else nn.Identity(),
            f,

            nn.ConvTranspose2d(args.ngf*4, args.ngf*2, 4, 2, 1, bias = not args.g_batchnorm),
            nn.BatchNorm2d(args.gnf*2) if args.g_batchnorm else nn.Identity(),
            f,

            nn.ConvTranspose2d(args.ngf*2, args.ngf*1, 4, 2, 1, bias = not args.g_batchnorm),
            nn.BatchNorm2d(args.ngf*1) if args.g_batchnorm else nn.Identity(),
            f,

            nn.ConvTranspose2d(args.ngf*1, args.nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.gen(z)


class _netE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        apply_sn = sn if args.e_sn else lambda x: x

        f = get_activation(args.e_activation, args)

        self.ebm = nn.Sequential(
            apply_sn(nn.Linear(args.nz, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, args.nez))
        )

    def forward(self, z):
        return self.ebm(z.squeeze()).view(-1, self.args.nez, 1, 1)


class _netWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.netG = _netG(args)
        self.netE = _netE(args)

        self.netG.apply(weights_init_xavier)
        self.netE.apply(weights_init_xavier)

    def sample_langevin_prior_z(self, z, netE, args, verbose=False):
        z = z.clone().detach()
        z.requires_grad = True
        for i in range(args.e_l_steps):
            en = netE(z)
            z_grad = torch.autograd.grad(en.sum(), z)[0]

            z.data = z.data - 0.5 * args.e_l_step_size * args.e_l_step_size * (z_grad + 1.0 / (args.e_prior_sig * args.e_prior_sig) * z.data)
            if args.e_l_with_noise:
                z.data += args.e_l_step_size * torch.randn_like(z).data

            if (i % 5 == 0 or i == args.e_l_steps - 1) and verbose:
                print('Langevin prior {:3d}/{:3d}: energy={:8.3f}'.format(i+1, args.e_l_steps, en.sum().item()))

            z_grad_norm = z_grad.view(args.batch_size, -1).norm(dim=1).mean()

        return z.detach(), z_grad_norm

    def sample_langevin_post_z(self, z, x, netG, netE, args, verbose=False):

        mse = nn.MSELoss(reduction='sum')

        z = z.clone().detach()
        z.requires_grad = True
        for i in range(args.g_l_steps):
            x_hat = netG(z)
            g_log_lkhd = 1.0 / (2.0 * args.g_llhd_sigma * args.g_llhd_sigma) * mse(x_hat, x)
            z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

            en = netE(z)
            z_grad_e = torch.autograd.grad(en.sum(), z)[0]

            z.data = z.data - 0.5 * args.g_l_step_size * args.g_l_step_size * (z_grad_g + z_grad_e + 1.0 / (args.e_prior_sig * args.e_prior_sig) * z.data)
            if args.g_l_with_noise:
                z.data += args.g_l_step_size * torch.randn_like(z).data

            if (i % 5 == 0 or i == args.g_l_steps - 1) and verbose:
                print('Langevin posterior {:3d}/{:3d}: MSE={:8.3f}'.format(i+1, args.g_l_steps, g_log_lkhd.item()))

            z_grad_g_grad_norm = z_grad_g.view(args.batch_size, -1).norm(dim=1).mean()
            z_grad_e_grad_norm = z_grad_e.view(args.batch_size, -1).norm(dim=1).mean()

        return z.detach(), z_grad_g_grad_norm, z_grad_e_grad_norm

    def forward(self, z, x=None, prior=True):
        # print('z', z.shape)
        # if x is not None:
        #    print('x', x.shape)

        if prior:
            return self.sample_langevin_prior_z(z, self.netE, self.args)[0]
        else:
            return self.sample_langevin_post_z(z, x, self.netG, self.netE, self.args)[0]


##########################################################################################################

def train(args_job, output_dir_job, output_dir, return_dict):

    #################################################
    ## preamble

    args = parse_args()
    args = pygrid.overwrite_opt(args, args_job)
    args = to_named_dict(args)

    # set_gpu(args.device)
    set_cuda(deterministic=args.gpu_deterministic)
    set_seed(args.seed)

    makedirs_exp(output_dir)

    job_id = int(args['job_id'])

    logger = setup_logging('job{}'.format(job_id), output_dir, console=True)
    logger.info(args)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    #################################################
    ## data

    ds_train, ds_val = get_dataset(args)
    logger.info('len(ds_train)={}'.format(len(ds_train)))
    logger.info('len(ds_val)={}'.format(len(ds_val)))

    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=True, num_workers=0)

    assert len(ds_train) >= args.n_fid_samples
    to_range_0_1 = lambda x: (x + 1.) / 2.
    # ds_fid = np.array(torch.stack([to_range_0_1(torch.tensor(ds_train[i][0])) for i in range(args.n_fid_samples)]).cpu().numpy())
    # logger.info('ds_fid.shape={}'.format(ds_fid.shape))
    ds_fid = []

    def plot(p, x):
        return torchvision.utils.save_image(torch.clamp(x, -1., 1.), p, normalize=True, nrow=int(np.sqrt(args.batch_size)))

    #################################################
    ## model

    if args.gpu_multi:
        net = torch.nn.DataParallel(_netWrapper(args).to(device), device_ids=[0,1,2,3])
    else:
        net = _netWrapper(args).to(device)

    def eval_flag():
        net.eval()

    def train_flag():
        net.train()

    def energy(score):
        if args.e_energy_form == 'tanh':
            energy = F.tanh(-score.squeeze())
        elif args.e_energy_form == 'sigmoid':
            energy = F.sigmoid(score.squeeze())
        elif args.e_energy_form == 'identity':
            energy = score.squeeze()
        elif args.e_energy_form == 'softplus':
            energy = F.softplus(score.squeeze())
        return energy

    mse = nn.MSELoss(reduction='sum')

    #################################################
    ## optimizer

    if args.gpu_multi:
        net_resolve = net.module
    else:
        net_resolve = net

    optE = torch.optim.Adam(net_resolve.netE.parameters(), lr=args.e_lr, weight_decay=args.e_decay, betas=(args.e_beta1, args.e_beta2))
    optG = torch.optim.Adam(net_resolve.netG.parameters(), lr=args.g_lr, weight_decay=args.g_decay, betas=(args.g_beta1, args.g_beta2))

    lr_scheduleE = torch.optim.lr_scheduler.ExponentialLR(optE, args.e_gamma)
    lr_scheduleG = torch.optim.lr_scheduler.ExponentialLR(optG, args.g_gamma)

    #################################################
    ## ckpt

    epoch_ckpt = 0

    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location='cuda:{}'.format(args.device))
        net_resolve.netE.load_state_dict(ckpt['netE'])
        optE.load_state_dict(ckpt['optE'])
        net_resolve.netG.load_state_dict(ckpt['netG'])
        optG.load_state_dict(ckpt['optG'])
        epoch_ckpt = 76


    #################################################
    ## sampling

    def sample_p_0(n=args.batch_size, sig=args.e_init_sig):
        return sig * torch.randn(*[n, args.nz, 1, 1]).to(device)

    #################################################
    ## fid

    def get_fid(n):

        assert n <= ds_fid.shape[0]

        logger.info('computing fid with {} samples'.format(n))

        try:
            eval_flag()

            def sample_x():
                z_0 = sample_p_0().to(device)
                z_k = net(Variable(z_0), prior=True)
                x_samples = to_range_0_1(net_resolve.netG(z_k)).clamp(min=0., max=1.).detach().cpu()
                return x_samples
            x_samples = torch.cat([sample_x() for _ in range(int(n / args.batch_size))]).numpy()
            fid = compute_fid_nchw(args, ds_fid[:n], x_samples)

            return fid

        except Exception as e:
            print(e)
            logger.critical(e, exc_info=True)
            logger.info('FID failed')

        finally:
            train_flag()

    # get_fid(n=args.batch_size)

    #################################################
    ## train

    train_flag()

    fid = 0.0
    fid_best = math.inf

    def normalize(x):
        return ((x.float() / 255.) - .5) * 2.

    z_fixed = sample_p_0()
    x_fixed = normalize(next(iter(dataloader_train))[0]).to(device)

    stats = {
        'loss_g':[],
        'loss_e':[],
        'en_neg':[],
        'en_pos':[],
        'grad_norm_g':[],
        'grad_norm_e':[],
        'z_e_grad_norm':[],
        'z_g_grad_norm':[],
        'z_e_k_grad_norm':[],
        'fid':[],
    }
    interval = []

    for epoch in range(epoch_ckpt, args.n_epochs):

        for i, (x, y) in enumerate(dataloader_train, 0):

            train_flag()

            x = normalize(x).to(device)
            batch_size = x.shape[0]

            # Initialize chains
            z_g_0 = sample_p_0(n=batch_size)
            z_e_0 = sample_p_0(n=batch_size)

            # Langevin posterior and prior
            z_g_k = net(Variable(z_g_0), x, prior=False)
            z_e_k = net(Variable(z_e_0), prior=True)

            # Learn generator
            optG.zero_grad()
            x_hat = net_resolve.netG(z_g_k.detach())
            loss_g = mse(x_hat, x) / batch_size
            loss_g.backward()
            # grad_norm_g = get_grad_norm(net.netG.parameters())
            # if args.g_is_grad_clamp:
            #    torch.nn.utils.clip_grad_norm(net.netG.parameters(), opt.g_max_norm)
            optG.step()

            # Learn prior EBM
            optE.zero_grad()
            en_neg = energy(net_resolve.netE(z_e_k.detach())).mean() # TODO(nijkamp): why mean() here and in Langevin sum() over energy? constant is absorbed into Adam adaptive lr
            en_pos = energy(net_resolve.netE(z_g_k.detach())).mean()
            loss_e = en_pos - en_neg
            loss_e.backward()
            # grad_norm_e = get_grad_norm(net.netE.parameters())
            # if args.e_is_grad_clamp:
            #    torch.nn.utils.clip_grad_norm_(net.netE.parameters(), args.e_max_norm)
            optE.step()

            # Printout
            if i % args.n_printout == 0:
                with torch.no_grad():
                    x_0 = net_resolve.netG(z_e_0)
                    x_k = net_resolve.netG(z_e_k)

                    en_neg_2 = energy(net_resolve.netE(z_e_k)).mean()
                    en_pos_2 = energy(net_resolve.netE(z_g_k)).mean()

                    prior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_e_k.mean(), z_e_k.std(), z_e_k.abs().max())
                    posterior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_g_k.mean(), z_g_k.std(), z_g_k.abs().max())

                    logger.info('{} {:5d}/{:5d} {:5d}/{:5d} '.format(job_id, epoch, args.n_epochs, i, len(dataloader_train)) +
                        'loss_g={:8.3f}, '.format(loss_g) +
                        'loss_e={:8.3f}, '.format(loss_e) +
                        'en_pos=[{:9.4f}, {:9.4f}, {:9.4f}], '.format(en_pos, en_pos_2, en_pos_2-en_pos) +
                        'en_neg=[{:9.4f}, {:9.4f}, {:9.4f}], '.format(en_neg, en_neg_2, en_neg_2-en_neg) +
                        '|z_g_0|={:6.2f}, '.format(z_g_0.view(batch_size, -1).norm(dim=1).mean()) +
                        '|z_g_k|={:6.2f}, '.format(z_g_k.view(batch_size, -1).norm(dim=1).mean()) +
                        '|z_e_0|={:6.2f}, '.format(z_e_0.view(batch_size, -1).norm(dim=1).mean()) +
                        '|z_e_k|={:6.2f}, '.format(z_e_k.view(batch_size, -1).norm(dim=1).mean()) +
                        'z_e_disp={:6.2f}, '.format((z_e_k-z_e_0).view(batch_size, -1).norm(dim=1).mean()) +
                        'z_g_disp={:6.2f}, '.format((z_g_k-z_g_0).view(batch_size, -1).norm(dim=1).mean()) +
                        'x_e_disp={:6.2f}, '.format((x_k-x_0).view(batch_size, -1).norm(dim=1).mean()) +
                        'prior_moments={}, '.format(prior_moments) +
                        'posterior_moments={}, '.format(posterior_moments) +
                        'fid={:8.2f}, '.format(fid) +
                        'fid_best={:8.2f}'.format(fid_best))

        # Schedule
        lr_scheduleE.step(epoch=epoch)
        lr_scheduleG.step(epoch=epoch)

        # Stats
        if epoch % args.n_stats == 0:
            stats['loss_g'].append(loss_g.item())
            stats['loss_e'].append(loss_e.item())
            stats['en_neg'].append(en_neg.data.item())
            stats['en_pos'].append(en_pos.data.item())
            stats['grad_norm_g'].append(0)
            stats['grad_norm_e'].append(0)
            stats['z_g_grad_norm'].append(0)
            stats['z_e_grad_norm'].append(0)
            stats['z_e_k_grad_norm'].append(0)
            stats['fid'].append(fid)
            interval.append(epoch + 1)
            plot_stats(output_dir, stats, interval)

        # Metrics
        if False and epoch % args.n_metrics == 0:
            fid = get_fid(n=len(ds_fid))
            if fid < fid_best:
                fid_best = fid
            logger.info('fid={}'.format(fid))

        # Plot
        if epoch % args.n_plot == 0:

            batch_size_fixed = x_fixed.shape[0]

            z_g_0 = sample_p_0(n=batch_size_fixed)
            z_e_0 = sample_p_0(n=batch_size_fixed)

            z_g_k = net(Variable(z_g_0), x_fixed)
            z_e_k = net(Variable(z_e_0), prior=True)

            with torch.no_grad():
                plot('{}/samples/{:>06d}_{:>06d}_x_fixed.png'.format(output_dir, epoch, i), x_fixed)
                plot('{}/samples/{:>06d}_{:>06d}_x_fixed_hat.png'.format(output_dir, epoch, i), net_resolve.netG(z_g_k))
                plot('{}/samples/{:>06d}_{:>06d}_x_z_neg_0.png'.format(output_dir, epoch, i), net_resolve.netG(z_e_0))
                plot('{}/samples/{:>06d}_{:>06d}_x_z_neg_k.png'.format(output_dir, epoch, i), net_resolve.netG(z_e_k))
                plot('{}/samples/{:>06d}_{:>06d}_x_z_fixed.png'.format(output_dir, epoch, i), net_resolve.netG(z_fixed))

        # Ckpt
        if epoch > 0 and epoch % args.n_ckpt == 0:
            save_dict = {
                'epoch': epoch,
                'net': net.state_dict(),
                'optE': optE.state_dict(),
                'optG': optG.state_dict(),
            }
            torch.save(save_dict, '{}/ckpt/ckpt_{:>06d}.pth'.format(output_dir, epoch))

        # Early exit
        if False and epoch > 10 and loss_g > 500:
            logger.info('early exit condition 1: epoch > 10 and loss_g > 500')
            return_dict['stats'] = {'fid_best': fid_best, 'fid': fid, 'mse': loss_g.data.item()}
            return

        if False and epoch > 20 and fid > 100:
            logger.info('early exit condition 2: epoch > 20 and fid > 100')
            return_dict['stats'] = {'fid_best': fid_best, 'fid': fid, 'mse': loss_g.data.item()}
            return

    return_dict['stats'] = {'fid_best': fid_best, 'fid': fid, 'mse': loss_g.data.item()}
    logger.info('done')



##########################################################################################################
## Metrics

from fid_v2_tf_cpu import fid_score

def is_xsede():
    import socket
    return 'psc' in socket.gethostname()


def compute_fid(args, x_data, x_samples, use_cpu=False):

    assert type(x_data) == np.ndarray
    assert type(x_samples) == np.ndarray

    # RGB
    assert x_data.shape[3] == 3
    assert x_samples.shape[3] == 3

    # NHWC
    assert x_data.shape[1] == x_data.shape[2]
    assert x_samples.shape[1] == x_samples.shape[2]

    # [0,255]
    assert np.min(x_data) > 0.-1e-4
    assert np.max(x_data) < 255.+1e-4
    assert np.mean(x_data) > 10.

    # [0,255]
    assert np.min(x_samples) > 0.-1e-4
    assert np.max(x_samples) < 255.+1e-4
    assert np.mean(x_samples) > 1.

    if use_cpu:
        def create_session():
            import tensorflow.compat.v1 as tf
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.0
            config.gpu_options.visible_device_list = ''
            return tf.Session(config=config)
    else:
        def create_session():
            import tensorflow.compat.v1 as tf
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            config.gpu_options.visible_device_list = str(args.device)
            return tf.Session(config=config)

    path = None if not is_xsede() else '/pylon5/ac5fpjp/bopang/ebm_prior/'

    fid = fid_score(create_session, x_data, x_samples, path, cpu_only=use_cpu)

    return fid

def compute_fid_nchw(args, x_data, x_samples):

    to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))

    x_data_nhwc = to_nhwc(255 * x_data)
    x_samples_nhwc = to_nhwc(255 * x_samples)

    fid = compute_fid(args, x_data_nhwc, x_samples_nhwc)

    return fid


##########################################################################################################
## Plots

import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_stats(output_dir, stats, interval):
    content = stats.keys()
    # f = plt.figure(figsize=(20, len(content) * 5))
    f, axs = plt.subplots(len(content), 1, figsize=(20, len(content) * 5))
    for j, (k, v) in enumerate(stats.items()):
        axs[j].plot(interval, v)
        axs[j].set_ylabel(k)

    f.savefig(os.path.join(output_dir, 'stat.pdf'), bbox_inches='tight')
    f.savefig(os.path.join(output_dir, 'stat.png'), bbox_inches='tight')
    plt.close(f)



##########################################################################################################
## Other

def get_grad_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def print_gpus():
    os.system('nvidia-smi -q -d Memory > tmp')
    tmp = open('tmp', 'r').readlines()
    for l in tmp:
        print(l, end = '')


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    free_gpu = np.argmax(memory_available)
    print('set gpu', free_gpu, 'with', np.max(memory_available), 'mb')
    return free_gpu


def set_gpu(gpu):
    torch.cuda.set_device('cuda:{}'.format(gpu))
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def to_named_dict(ns):
    d = AttrDict()
    for (k, v) in zip(ns.__dict__.keys(), ns.__dict__.values()):
        d[k] = v
    return d


def merge_dicts(a, b, c):
    d = {}
    d.update(a)
    d.update(b)
    d.update(c)
    return d


##########################################################################################################
## Main

def makedirs_exp(output_dir):
    os.makedirs(output_dir + '/samples')
    os.makedirs(output_dir + '/ckpt')

def main():

    print_gpus()

    fs_prefix = './'

    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = pygrid.get_output_dir(exp_id, fs_prefix=fs_prefix)

    # run
    copy_source(__file__, output_dir)
    opt = {'job_id': int(0), 'status': 'open', 'device': 0}
    train(opt, output_dir, output_dir, {})


if __name__ == '__main__':
    main()
