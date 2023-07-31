"""Train all models."""

from offline.offline_train import HIVES
from utilities.utils import params_extraction, load_pretrained_models
from utilities.optimization import set_optimizers
from rl.agent import VaLS
from rl.sampler import Sampler, NormalReplayBuffer, ModifiedReplayBuffer
from datetime import datetime
from models.nns import Critic, SkillPolicy, StateEncoder, StateDecoder
import wandb
import os
import torch
import numpy as np
import copy
import pickle

torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=5)

import pdb

os.environ['WANDB_SILENT'] = "true"

wandb.login()

ENV_NAME = 'relocate-expert-v1'
PARENT_FOLDER = 'checkpoints_relocate'
CASE_FOLDER = 'New_Baseline'

config = {
    # General hyperparams
    'device': 'cuda',
    'hidden_dim': 128,
    'env_id': ENV_NAME,
    
    # Offline hyperparams
    'vae_batch_size': 1024,
    'vae_lr': 6e-4,
    'priors_lr': 6e-4,
    'epochs': 501,
    'beta': 0.01,
    'length': 8,
    'z_skill_dim': 12,

    # Online hyperparams  
    'batch_size': 256,
    'action_range': 4,
    'critic_lr': 3e-4,
    'actor_lr': 3e-4,
    'discount': 0.97,
    'delta_skill': 32,
    'delta_length': 32,
    'z_state_dim': 8,
    'gradient_steps': 8,
    'max_iterations': int(200000 + 1),
    'buffer_size': int(200000 + 1),
    'test_freq': 100000,
    'reset_frequency': 300,
    'singular_val_k': 1,

    # Run params
    'train_VAE_models': False,
    'train_priors': False,
    'train_rl': True,
    'load_VAE_models': True,
    'load_prior_models': True,
    'load_rl_models': True,
    'use_SAC': False,
    'modified_buffer': True,
    'render_results': False
}


path_to_data = f'datasets/{ENV_NAME}.pt'


def main(config=None):
    """Train all modules."""
    with wandb.init(project='ReplayBuffer-Relocate-(SVDvals)', config=config,
                    notes='This logs singular values',
                    name='Test'):

        config = wandb.config

        path = PARENT_FOLDER
        hives = HIVES(config)

        if not config.train_rl:
            hives.dataset_loader(path_to_data)

        skill_policy = SkillPolicy(hives.state_dim, hives.action_range,
                                   latent_dim=hives.z_skill_dim).to(hives.device)

        critic = Critic(hives.state_dim, hives.z_skill_dim).to(hives.device)
        
        sampler = Sampler(skill_policy, hives.evaluate_decoder, config)

        if config.modified_buffer:
            experience_buffer = ModifiedReplayBuffer(hives.buffer_size, sampler.env, hives.z_skill_dim, config.reset_frequency)
        else:
            experience_buffer = NormalReplayBuffer(hives.buffer_size, sampler.env, hives.z_skill_dim)

        with open('checkpoints_relocate/class', 'rb') as file:
            aux_vals = pickle.load(file)

        def get_zero_obs_idxs(obs):
            obs_init = obs[:, 0:30]
            return np.argwhere(obs_init.sum(1) == 0).squeeze()
        
        idxs_offline = get_zero_obs_idxs(aux_vals.experience_buffer.offline_obs_buf)
        idxs_online = get_zero_obs_idxs(aux_vals.experience_buffer.obs_buf[0:25000,:])

        import matplotlib.pyplot as plt
        obs_off = aux_vals.experience_buffer.offline_obs_buf[idxs_offline][:, 30:]
        obs_on = aux_vals.experience_buffer.obs_buf[idxs_online][:,30:]

        rews_on = aux_vals.experience_buffer.cum_reward[idxs_online].squeeze()

        def comparison(obs1, obs2):
            diff = np.abs(obs1.reshape(-1, 1, 9) - obs2.reshape(1, -1, 9)).max(2)
            return diff
            

        off_on = comparison(obs_off, obs_on)
        off_off = comparison(obs_off, obs_off)
        on_on = comparison(obs_on, obs_on)

        sorted_off_on = np.sort(comparison(obs_off, obs_on), 0)
        sorted_off_off = np.sort(comparison(obs_off, obs_off), 0)
        sorted_on_on = np.sort(comparison(obs_on, obs_on), 0)
        
        good_runs = obs_on[rews_on > 15]
        bad_runs = obs_on[rews_on < .5]

        runs = np.concatenate((good_runs, bad_runs), axis=0)

        on_good = comparison(obs_on, good_runs)
        on_bad = comparison(obs_on, bad_runs)

        on_good_0 = np.sort(on_good, 0)
        on_bad_0 = np.sort(on_bad, 0)

        on_good_1 = np.sort(on_good_0, 1)
        on_bad_1 = np.sort(on_bad_0, 1)
        
        
            
        
        pdb.set_trace()

        # exp_idx = 100000
        # idx = 300000
        # experience_buffer.obs_buf[0: exp_idx, :] = aux_vals.experience_buffer.obs_buf[idx: idx + exp_idx, :]
        # experience_buffer.next_obs_buf[0: exp_idx, :] = aux_vals.experience_buffer.next_obs_buf[idx: idx + exp_idx, :]
        # experience_buffer.z_buf[0: exp_idx, :] = aux_vals.experience_buffer.z_buf[idx: idx + exp_idx, :]
        # experience_buffer.next_z_buf[0: exp_idx, :] = aux_vals.experience_buffer.next_z_buf[idx: idx + exp_idx, :]
        # experience_buffer.rew_buf[0: exp_idx, :] = aux_vals.experience_buffer.rew_buf[idx: idx + exp_idx, :]
        # experience_buffer.done_buf[0: exp_idx, :] = aux_vals.experience_buffer.done_buf[idx: idx + exp_idx, :]
        # experience_buffer.cum_reward[0: exp_idx, :] = aux_vals.experience_buffer.cum_reward[idx: idx + exp_idx, :]
        
        # experience_buffer.ptr = exp_idx
        # experience_buffer.size = exp_idx

        vals = VaLS(sampler,
                    experience_buffer,
                    hives,
                    skill_policy,
                    critic,
                    config)
        
        hives_models = list(hives.models.values())

        models = [*hives_models, vals.skill_policy,
                  vals.critic, vals.critic,
                  vals.critic, vals.critic]
        
        names = [*hives.names, 'SkillPolicy', 'Critic1', 'Target_critic1',
                 'Critic2', 'Target_critic2']
    
        params_path = 'params_rl.pt'
        
        pretrained_params = load_pretrained_models(config, PARENT_FOLDER, params_path)
        pretrained_params.extend([None] * (len(names) - len(pretrained_params)))
        
        params = params_extraction(models, names, pretrained_params)

        vals.experience_buffer.log_offline_dataset('datasets/relocate-expert-v1.pt',
                                                   params, hives.evaluate_encoder, hives.device)
        
        test_freq = config.epochs // 4
        test_freq = test_freq if test_freq > 0 else 1
    
        keys_optims = ['VAE_skills']
        keys_optims.extend(['SkillPrior', 'SkillPolicy'])
        keys_optims.extend(['Critic1', 'Critic2'])

        optimizers = set_optimizers(params, keys_optims, config.critic_lr)

        print('Training is starting')
    
        if config.train_VAE_models:
            for e in range(config.epochs):
                params = hives.train_vae(params,
                                         optimizers,
                                         config.vae_lr,
                                         config.beta)
                if e % test_freq == 0:
                    print(f'Epoch is {e}')
                    dt = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(params, f'{path}/params_{dt}_epoch{e}.pt')
                
        if config.train_priors:
            hives.set_skill_lookup(params)
            for i in range(config.epochs):
                params = hives.train_prior(params, optimizers,
                                           config.priors_lr)

            folder = 'Prior'
            dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            fullpath = f'{path}/{folder}'
            if not os.path.exists(fullpath):
                os.makedirs(fullpath)
                
            torch.save(params, f'{path}/{folder}/params_{dt_string}_offline.pt')

        if config.train_rl:
            params = vals.training(params, optimizers, path, CASE_FOLDER)

        if config.render_results:
            vals.render_results(params, f'videos/{config.env_id}/{CASE_FOLDER}')


            
main(config=config)

