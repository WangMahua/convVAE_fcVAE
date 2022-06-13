import argparse
import itertools
import os
import random
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from PIL import Image

from dataset_image import mimii_dataset
# from models.lstm import gaussian_lstm, lstm
# from models.vgg_64 import vgg_decoder, vgg_encoder
from models.vae import ConvVAE,CVAE
from util import *

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='valve')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data_image', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=10, help='epoch size')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--beta', type=float, default=0.001, help='weighting on KL to prior')
    parser.add_argument('--cuda', default=True, action='store_true') 

    args = parser.parse_args()
    return args

mse_criterion = nn.MSELoss()


def loss_function( args,model,reconstructed_x, x, mu, logvar, use_bce=True):
    """
    Reconstruction + KL divergence losses summed over all elements and
    batch.
    If for the reconstruction loss, the binary cross entropy is used, be
    sure to have an adequate activation for the last layer of the decoder
    (eg a sigmoid). Same goes if binary cross entropy is not used (in that
    case, mean squared error is used, you could use a tanh activation)
    """
    if use_bce:
        reconstruction_loss = F.binary_cross_entropy(
            model.flatten(reconstructed_x),
            model.flatten(x),
            reduction='sum')
    else:
        reconstruction_loss = F.mse_loss(
            model.flatten(reconstructed_x),
            model.flatten(x),
            reduction='sum')

    # Adding a beta value for a beta VAE. With beta = 1, standard VAE
    # beta = 1.0

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5* torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * args.beta

    return reconstruction_loss + KLD


def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        name = '{}'.format(nowTime)
        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    os.makedirs(args.log_dir, exist_ok=True)
    # os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --------- load a dataset ------------------------------------
    train_dataset = mimii_dataset(args,'train')
    # val_dataset = mimii_dataset(args, 'validate')
    train_loader = torch.utils.data.DataLoader(train_dataset,\
                        batch_size=args.batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset,\
                        # batch_size=args.batch_size, shuffle=True)

    train_iterator = iter(train_loader)
    # validate_iterator = iter(val_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    # --------- training loop ------------------------------------

    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    # model = ConvVAE(channels=3, z_dim=128).to(device)
    model = CVAE(args.batch_size,channels=3, z_dim=32).to(device)
    optimizer = args.optimizer(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))


    for epoch in range(args.niter):

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0
        train_loss = 0.0

        model.train()
        for epoch_ in range(args.epoch_size):
            try:
                seq = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq = next(train_iterator)

            seq = seq.type(torch.FloatTensor)
            seq = seq.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(seq)

            loss = loss_function(args,model,
                recon_batch,
                seq,
                mu,
                logvar,
                use_bce=False)
            a = list(seq.shape)
            loss = loss/a[0]
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        

        # Save image
        # recon_seq, _, _ = model(seq)
        compare_seq = torch.cat([seq, recon_batch])
        save_image(compare_seq.data.cpu(), args.log_dir + f'/recon_image_{epoch}.png')

        progress.update(1)

        avg_epoch_loss = epoch_loss/(args.epoch_size*len(train_loader))

        if epoch == 0:
            best_loss = avg_epoch_loss

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model, args.log_dir + '/model.pt')

        print("\nLoss: {:.3f} | Best_loss: {:.3f}".format(avg_epoch_loss, best_loss))
        
        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('epoch: %03d | loss: %.3f (best loss: %.3f) \n'  % (epoch+1, best_loss, avg_epoch_loss)))
        model.eval()


if __name__ == '__main__':
    main()
        
