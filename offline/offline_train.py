"""Define all offline training, e.g., decoder, encoder priors."""

import sys
sys.path.insert(0, '../')

import os
import numpy as np
import torch
from torch.func import functional_call
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utilities.optimization import Adam_update
from utilities.utils import hyper_params
from models.nns import SkillPrior, LengthPrior
from models.nns import Encoder, Decoder
import wandb
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class HIVES(hyper_params):
    """Define all offline and train all offline models."""
    def __init__(self, config):
        """Config contain all VAE hyper parameters."""
        super().__init__(config)
    
        self.models = {}
        self.names = []

        encoder = Encoder(self.length,
                          input_dim=self.action_dim,
                          latent_dim=self.z_skill_dim).to(self.device)

        decoder = Decoder(self.length,
                          output_dim=self.action_dim,
                          input_dim=self.z_skill_dim).to(self.device)

        self.models['Encoder'] = encoder
        self.models['Decoder'] = decoder

        self.names.extend(['Encoder', 'Decoder'])

        self.models['SkillPrior'] = SkillPrior(
            self.state_dim, latent_dim=self.z_skill_dim).to(self.device)

        self.names.extend(['SkillPrior'])

    def dataset_loader(self, path):
        """Dataset is one single file."""
        self.load_dataset(path)
        
        dset_train = Drivedata(self.indexes)

        self.loader = DataLoader(dset_train, shuffle=True, num_workers=8,
                                 batch_size=self.vae_batch_size)

    def train_vae(self, params, optimizers, lr, beta):
        for i, idx in enumerate(self.loader):
            action = torch.from_numpy(self.dataset['actions'][idx])
            action = action.view(-1, action.shape[-1]).to(self.device)
            recon_loss, kl_loss = self.vae_loss(action, params, i)
            loss = recon_loss + beta * kl_loss
            losses = [loss]
            params_names = ['VAE_skills']
            params = Adam_update(params, losses, params_names, optimizers, lr)

        wandb.log({'VAE/loss': recon_loss.detach().cpu()})
        wandb.log({'VAE/kl_loss': kl_loss.detach().cpu()})
        
        return params

    def vae_loss(self, action, params, i):
        """VAE loss."""
        z_seq, pdf, mu, std = self.evaluate_encoder(action, params)
        rec = self.evaluate_decoder(z_seq, params)
        
        error = torch.square(action - rec).mean(1)
        rec_loss = -Normal(rec, 1).log_prob(action).sum(axis=-1)
        rec_loss = rec_loss.mean()

        if i == 0:
            wandb.log({'VAE/[encoder] STD':
                       wandb.Histogram(std.detach().cpu())})
            wandb.log({'VAE/[encoder] Mu':
                       wandb.Histogram(mu.detach().cpu())})
            wandb.log({'VAE/[decoder]reconstruction_std': rec.std(0).mean().detach().cpu()})
            if rec_loss < 5:
                wandb.log({'VAE/MSE Distribution':
                           wandb.Histogram(error.detach().cpu())})

        N = Normal(0, 1)
        kl_loss = torch.mean(kl_divergence(pdf, N))

        return rec_loss, kl_loss

    def evaluate_encoder(self, mu, params):
        """Evaluate encoder."""
        mu = mu.reshape(-1, self.length, mu.shape[-1])
        z, pdf, mu, std = functional_call(self.models['Encoder'],
                                          params['Encoder'], mu)

        return z, pdf, mu, std

    def evaluate_decoder(self, rec, params):
        """Evaluate decoder. """
        rec = functional_call(self.models['Decoder'],
                              params['Decoder'], rec)
        rec = rec.reshape(-1, rec.shape[-1])
            
        return rec
        
    def train_prior(self, params, optimizers, lr, length=True):
        """Trains one epoch of length prior."""
        for i, idx in enumerate(self.loader):
            obs = self.dataset['observations'][idx][:, 0, :]
            obs = torch.from_numpy(obs).to(self.device)
            prior_loss = self.skill_prior_loss(idx, obs, params, i)
            name = ['SkillPrior']
            loss = [prior_loss]
            params = Adam_update(params, loss, name, optimizers, lr)

        return params
    
    def skill_prior_loss(self, idx, obs, params, i):
        """Compute loss for skill prior."""

        prior = functional_call(self.models['SkillPrior'],
                                params['SkillPrior'],
                                obs)

        pdf = Normal(self.loc[idx, :], self.scale[idx, :])

        kl_loss = kl_divergence(prior, pdf)

        if i == 0:
            wandb.log({'skill_prior/KL loss': kl_loss.mean().detach().cpu()})
            wandb.log({'skill_prior/min std': pdf.scale.min().detach().cpu()})
            wandb.log({'skill_prior/max std': pdf.scale.max().detach().cpu()})
            if kl_loss.mean() < .2:
                wandb.log({'skill_prior/KL dist':
                           wandb.Histogram(kl_loss.detach().cpu())})
            wandb.log({'skill_prior/VAE std dist':
                       wandb.Histogram(pdf.scale.detach().cpu())})
            wandb.log({'skill_prior/VAE mu dist':
                       wandb.Histogram(pdf.loc.detach().cpu())})
            
        return kl_loss.mean()

    def set_skill_lookup(self, params):
        """Save all skils to train priors."""
        all_actions = torch.from_numpy(self.dataset['actions']).to(self.device)
        a, b, _ = all_actions.shape
        
        self.loc = torch.zeros(a, self.z_skill_dim).to(self.device)
        self.scale = torch.zeros(a, self.z_skill_dim).to(self.device)

        bs_size = 1024
        number_of_batches = all_actions.shape[0] // bs_size + 1

        for j in range(number_of_batches):
            actions = all_actions[j * bs_size:(j + 1) * bs_size, :, :]
            with torch.no_grad():
                _, pdf, z, _ = self.evaluate_encoder(actions, params)
            loc, scale = pdf.loc, pdf.scale
            self.loc[j * bs_size: (j + 1) * bs_size, :] = loc
            self.scale[j * bs_size: (j + 1) * bs_size, :] = scale

                
    def load_dataset(self, path):
        """Extract sequences of length max_length from episodes.

        This dataset requires to know when terminal states occur.
        """
        data = torch.load(path)

        keys = ['actions', 'observations']
        dataset = {}
        self.max_length = self.length

        terminal_key = 'terminals' if 'kitchen' in self.env_id else 'timeouts'
        # 'terminals' for kitchen environment; 'timeouts' for adroit.

        terminal_idxs = np.arange(len(data[terminal_key]))
        terminal_idxs = terminal_idxs[data[terminal_key]]
        
        episode_cutoff = terminal_idxs[0] + self.max_length

        self.run = (terminal_idxs[0],
                    data['actions'][:self.max_length * (episode_cutoff // self.max_length)])
        
        idxs = []
        base_idx = np.arange(self.max_length)
        old_idx = 0
        
        for idx in terminal_idxs:
            samples = idx - old_idx - self.max_length
            if samples < 0:
                continue
            new_idx = np.repeat(base_idx[np.newaxis, :], samples, 0)
            new_idx = new_idx + np.arange(samples)[:, np.newaxis] + old_idx
            idxs.extend(new_idx.flatten())
            old_idx = idx
            
        for key in keys:
            val_dim = data[key].shape[-1] if len(data[key].shape) > 1 else 1
            seqs = np.take(data[key], idxs, axis=0)
            seqs = seqs.reshape(-1, self.max_length, val_dim).squeeze()
            dataset[key] = seqs

        self.dataset = dataset
        self.indexes = torch.arange(self.dataset['actions'].shape[0])       

class Drivedata(Dataset):
    """Dataset loader."""

    def __init__(self, indexes, transform=None):
        """Dataset init."""
        self.xs = indexes

    def __getitem__(self, index):
        """Get given item."""
        return self.xs[index]

    def __len__(self):
        """Compute length."""
        return len(self.xs)
