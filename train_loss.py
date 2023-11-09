import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
import argparse
from Models import Vae

# training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create our model and send it to the device (cpu/gpu)
# vae = Vae(x_dim=X_DIM, z_dim=Z_DIM, hidden_size=HIDDEN_SIZE, device=device).to(device)
# # optimizer
# vae_optim = torch.optim.Adam(params=vae.parameters(), lr=LEARNING_RATE)
# save the losses from each epoch, we might want to plot it later
def train(dataloader, args):
  train_losses = []
  train_kl_losses = []
  train_recon_losses = []
  # here we go
  for beta in [0.5,1.5,2]:
    print("The beta is:",beta)
    vae = Vae(x_dim=args.x_dim, z_dim=args.z_dim, hidden_size=args.hidden_size, device=device, num_layers=args.trans_layers, extraDecFC_dim=args.extra_dec_dim, isExtraLayer=args.extraDec).to(device)
    # optimizer
    vae_optim = torch.optim.Adam(params=vae.parameters(), lr=args.lr)
    for epoch in range(args.n_epochs):
      epoch_start_time = time.time()
      batch_losses = []
      batch_kl = []
      batch_recon = []
      for batch_i, batch in enumerate(dataloader):
        # forward pass
        x = batch.to(device).view(-1, args.x_dim) # just the images
        x_recon, mu, logvar, z = vae(x)
        # calculate the loss
        x = x.float()
        loss,kl_d,recon_err = beta_loss_function(x_recon, x, mu, logvar, loss_type=args.recon_loss, beta=beta)
        #print(type(loss))
        # optimization (same 3 steps everytime)
        vae_optim.zero_grad()
        loss.backward()
        vae_optim.step()
        # save loss
        batch_losses.append(loss.data.cpu().item())
        batch_kl.append(kl_d)
        batch_recon.append(recon_err)
      train_losses.append(np.mean(batch_losses))
      train_kl_losses.append(np.mean(batch_kl))
      train_recon_losses.append(np.mean(batch_recon))
      print("epoch: {} training loss: {:.5f} KL loss: {:.5f} Reconstruction loss {:.5f} epoch time: {:.3f} sec".format(epoch, train_losses[-1],
              train_kl_losses[-1], train_recon_losses[-1],time.time() - epoch_start_time))

    # save
    check_pts_names = []
    check_pts_names.append(args.exp_dir + "/beta_" + str(beta) + ".pth")
    torch.save(vae.state_dict(), check_pts_names[-1])
    print("saved checkpoint @", check_pts_names[-1])

  return train_losses, train_kl_losses, train_recon_losses

def loss_function(recon_x, x, mu, logvar, loss_type='bce'):
  """
  This function calculates the loss of the VAE.
  loss = reconstruction_loss - 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  :param recon_x: the reconstruction from the decoder
  :param x: the original input
  :param mu: the mean given X, from the encoder
  :param logvar: the log-variance given X, from the encoder
  :param loss_type: type of loss function - 'mse', 'l1', 'bce'
  :return: VAE loss
  """
  if loss_type == 'mse':
    recon_error = F.mse_loss(recon_x, x, reduction='sum')
  elif loss_type == 'l1':
    recon_error = F.l1_loss(recon_x, x, reduction='sum')
  elif loss_type == 'bce':
    recon_error = F.binary_cross_entropy(recon_x, x, reduction='sum')
  else:
    raise NotImplementedError

  # see Appendix B from VAE paper:
  # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
  # https://arxiv.org/abs/1312.6114
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return (recon_error + kl) / x.size(0)


def beta_loss_function(recon_x, x, mu, logvar, loss_type='bce',beta=1):
    """
    This function calculates the loss of the VAE.
    loss = reconstruction_loss - 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param recon_x: the reconstruction from the decoder
    :param x: the original input
    :param mu: the mean given X, from the encoder
    :param logvar: the log-variance given X, from the encoder
    :param loss_type: type of loss function - 'mse', 'l1', 'bce'
    :return: VAE loss
    """
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='sum')
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction='sum')
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        raise NotImplementedError

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -1*beta*0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_d = kl/beta
    return (recon_error + kl) / x.size(0), kl_d.data.cpu().numpy() / x.size(0),recon_error.data.cpu().numpy() / x.size(0)

