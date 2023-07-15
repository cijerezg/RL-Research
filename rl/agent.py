"""Training RL algorithm."""

import sys
sys.path.insert(0, '../')

import torch
from utilities.optimization import GD_full_update, Adam_update
from utilities.utils import hyper_params, process_frames, reset_params
from torch.func import functional_call
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import wandb
import numpy as np
from datetime import datetime
from torch.optim import Adam
import torch.autograd as autograd
import os
import pdb
import pickle
import time
from stable_baselines3.common.utils import polyak_update
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

MAX_SKILL_KL = 100
INIT_LOG_ALPHA = 0

class VaLS(hyper_params):
    def __init__(self,
                 sampler,
                 experience_buffer,
                 vae,
                 skill_policy,
                 critic,
                 args):

        super().__init__(args)

        self.sampler = sampler
        self.critic = critic
        self.skill_policy = skill_policy
        self.vae = vae
        self.experience_buffer = experience_buffer
        
        self.log_alpha_skill = torch.tensor(INIT_LOG_ALPHA, dtype=torch.float32,
                                            requires_grad=True,
                                            device=self.device)
        self.optimizer_alpha_skill = Adam([self.log_alpha_skill], lr=args.actor_lr)

        self.reward_per_episode = 0
        self.total_episode_counter = 0
        self.reward_logger = [0]
        self.log_data = 0
        self.log_data_freq = 1000
        self.email = True
        
    def training(self, params, optimizers, path, name):               
        self.iterations = 0
        ref_params = copy.deepcopy(params)

        obs = None    # These two lines are to setup the RL env.
        done = False  # Only need to be called once.

        while self.iterations < self.max_iterations:

            params, obs, done = self.training_iteration(params, done,
                                                        optimizers,
                                                        self.actor_lr,
                                                        ref_params,
                                                        obs=obs)

            if self.iterations % self.test_freq == 0 and self.iterations > 0:
                dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
                print(f'Current iteration is {self.iterations}')
                print(dt_string)
                fullpath = f'{path}/{name}'
                if not os.path.exists(fullpath):
                    os.makedirs(fullpath)
                filename = f'{path}/{name}/params_rl_{dt_string}_iter{self.iterations}.pt'
                torch.save(params, filename)
                with open(f'{fullpath}/class_{dt_string}_{self.iterations}', 'wb') as file:
                    pickle.dump(self, file)

            if self.iterations % self.log_data_freq == 0:
                wandb.log({'Iterations': self.iterations})
            self.iterations += 1

            if self.iterations == int(self.reset_frequency * 3 / 4):
                ref_params = copy.deepcopy(params)

            if self.iterations == self.reset_frequency:
                self.reset_frequency = 2 * self.reset_frequency
                self.gradient_steps = 1
                keys = ['SkillPolicy', 'Critic1', 'Critic2']
                ref_params = copy.deepcopy(params)
                params, optimizers = reset_params(params, keys, optimizers, self.actor_lr)
                self.log_alpha_skill = torch.tensor(INIT_LOG_ALPHA, dtype=torch.float32,
                                                    requires_grad=True,
                                                    device=self.device)
                self.optimizer_alpha_skill = Adam([self.log_alpha_skill], lr=self.actor_lr)

                self.experience_buffer.idx_tracker[:] = 0
                
        return params

    def training_iteration(self,
                           params,
                           done,
                           optimizers,
                           lr,
                           ref_params,
                           obs=None):
               
        obs, data = self.sampler.skill_iteration(params, done, obs)

        next_obs, rew, z, next_z, done = data

        if self.env_key == 'kitchen':
            self.reward_per_episode += rew
        elif self.env_key == 'adroit':
            self.reward_per_episode = rew

        self.experience_buffer.add(obs, next_obs, z, next_z, rew, done)

        if done:
            if self.total_episode_counter > 2 and self.modified_buffer:
                self.experience_buffer.update_tracking_buffers(self.reward_per_episode)

            wandb.log({'Reward per episode': self.reward_per_episode,
                       'Total episodes': self.total_episode_counter})

            self.reward_logger.append(self.reward_per_episode)
            self.reward_per_episode = 0
            self.total_episode_counter += 1

        log_data = True if self.log_data % self.log_data_freq == 0 else False

        if len(self.reward_logger) > 25 and log_data:
            wandb.log({'Cumulative reward dist': wandb.Histogram(np.array(self.reward_logger))})
            wandb.log({'Average reward over 100 eps': np.mean(self.reward_logger[-100:])}, step=self.iterations)

        self.log_data = (self.log_data + 1) % self.log_data_freq

        if self.experience_buffer.size > self.batch_size:

            for i in range(self.gradient_steps):
                policy_losses, critic1_loss, critic2_loss = self.losses(params, log_data, ref_params)
                losses = [*policy_losses, critic1_loss, critic2_loss]
                names = ['SkillPolicy', 'Critic1', 'Critic2']
                params = Adam_update(params, losses, names, optimizers, lr)
                polyak_update(params['Critic1'].values(),
                              params['Target_critic1'].values(), 0.005)
                polyak_update(params['Critic2'].values(),
                              params['Target_critic2'].values(), 0.005)

            if log_data:
                with torch.no_grad():
                    dist_init1 = self.distance_to_params(params, ref_params, 'Critic1', 'Critic1')
                    dist_init_pol = self.distance_to_params(params, ref_params, 'SkillPolicy', 'SkillPolicy')
                    
                wandb.log({'Critic/Distance to init weights': dist_init1,
                           'Policy/Distance to init weights Skills': dist_init_pol}) 
           
        return params, next_obs, done

    def losses(self, params, log_data, ref_params):
        batch = self.experience_buffer.sample(batch_size=self.batch_size)

        obs = torch.from_numpy(batch.observations).to(self.device)
        next_obs = torch.from_numpy(batch.next_observations).to(self.device)
        z = torch.from_numpy(batch.z).to(self.device)
        next_z = torch.from_numpy(batch.next_z).to(self.device)
        rew = torch.from_numpy(batch.rewards).to(self.device)
        dones = torch.from_numpy(batch.dones).to(self.device)
        cum_reward = torch.from_numpy(batch.cum_reward).to(self.device)
        idxs = torch.from_numpy(batch.idxs).to(self.device)

        if log_data:
            # # Critic analysis
            critic_test_arg = torch.cat([obs, z], dim=1)

            trials = 32

            new_z = z.reshape(z.shape[0], 1, -1).repeat(1, trials, 1)
            new_z = new_z.reshape(-1, new_z.shape[-1])
            z_rand = torch.rand(new_z.shape).to(self.device)
            new_z = new_z + torch.randn(new_z.shape).to(self.device) / 5

            new_obs = obs.reshape(obs.shape[0], 1, -1).repeat(1, trials, 1)
            new_obs = new_obs.reshape(-1, new_obs.shape[-1])

            new_critic_arg = torch.cat([new_obs, new_z], dim=1)
            new_critic_arg_rand = torch.cat([new_obs, z_rand], dim=1)
        
            with torch.no_grad():
                q1_r, q2_r = self.eval_critic(critic_test_arg, params)
                
                q1_rep, q2_rep = self.eval_critic(new_critic_arg, params)
                q1_rep, q2_rep = q1_rep.reshape(-1, trials, 1), q2_rep.reshape(-1, trials, 1)

                q1_rand, _ = self.eval_critic(new_critic_arg_rand, params)
                q1_rand = q1_rand.reshape(-1, trials, 1)
                
                max_diff1 = q1_r - q1_rep.max(1)[0]
                mean_diff1 = q1_r - q1_rep.mean(1)

                max_diff_rand = q1_r - q1_rand.max(1)[0]
                mean_diff_rand = q1_r - q1_rand.mean(1)

            eval_test_ave = self.log_scatter_3d(q1_r, q1_rand.max(1)[0], cum_reward, rew,
                                                'Q val', 'Q random', 'Cum reward', 'Reward')
            eval_test_max = self.log_scatter_3d(q1_r, q1_rep.max(1)[0], q1_rand.max(1)[0], idxs.unsqueeze(dim=1),
                                                'Q val', 'Q random centered', 'Q random', 'Idxs')

            wandb.log({'Critic/Max diff dist': wandb.Histogram(max_diff1.cpu()),
                       'Critic/Mean diff dist': wandb.Histogram(mean_diff1.cpu()),
                       'Critic/Max diff average': max_diff1.mean().cpu(),
                       'Critic/Mean diff average': mean_diff1.mean().cpu(),
                       'Critic/Max diff dist rand': wandb.Histogram(max_diff_rand.cpu()),
                       'Critic/Mean diff dist rand': wandb.Histogram(mean_diff_rand.cpu()),
                       'Critic/Max diff average rand': max_diff_rand.mean().cpu(),
                       'Critic/Mean diff average rand': mean_diff_rand.mean().cpu(),
                       'Policy/Eval policy critic_random': eval_test_ave,
                       'Policy/Eval policy max': eval_test_max
                       })

            bins = np.linspace(0, 400000, num=81)
            vals = []
            for i in range(bins.shape[0] - 1):
                aux = self.experience_buffer.idx_tracker[i* 5000: 5000*(i + 1)].sum()
                vals.append(aux)

            vals = np.array(vals)
            hist = (vals, bins)

            wandb.log({'Logged idx': wandb.Histogram(np_histogram=hist)})
                                                                 
        ####

        target_critic_arg = torch.cat([next_obs, next_z], dim=1)
        
        with torch.no_grad():                                
            z_prior = self.eval_skill_prior(obs, params)
            
            q_target1, q_target2 = self.eval_critic(target_critic_arg, params,
                                                    target_critic=True)

        q_target = torch.cat((q_target1, q_target2), dim=1)
        q_target, _ = torch.min(q_target, dim=1)
            
        critic_arg = torch.cat([obs, z], dim=1)

        q1, q2 = self.eval_critic(critic_arg, params)
        
        if log_data:
            with torch.no_grad():
                dist1 = self.distance_to_params(params, params, 'Critic1', 'Target_critic1')

                q1_next, _ = self.eval_critic(target_critic_arg, params)

                q_refs, _ = self.eval_critic(critic_arg, ref_params)
                q_refs_target, _ = self.eval_critic(target_critic_arg, ref_params,
                                                    target_critic=True)

            heatmap_Q_next = self.log_histogram_2d(q1, q1_next, 'Q val', 'Q next')
            heatmap_Qs = self.log_histogram_2d(q1, q_target.unsqueeze(dim=1), 'Q val', 'Q target')
            heatmap_Qref = self.log_histogram_2d(q1, q_refs, 'Q val', 'Q val ref')
            heatmap_Qreftarget = self.log_histogram_2d(q_target.unsqueeze(dim=1), q_refs_target, 'Q target', 'Q target ref')
            bellman_terms = self.log_scatter_3d(q1, q_target.unsqueeze(dim=1), rew, cum_reward,
                                                'Q val', 'Q target', 'Reward', 'Cum reward')
            bellman_terms_ref = self.log_scatter_3d(q1, q_refs, cum_reward, idxs.unsqueeze(dim=1),
                                                    'Q val', 'Q refs', 'Cum reward', 'Idxs')

            d1, d2, d3 = self.distance_to_target_env(obs)

            eval_crit = self.log_scatter_3d(d1, d2, d3, q1, 'Palm to ball', 'Palm to target',
                                         'Ball to target', 'Qval')

            
            wandb.log({'Critic/Distance critic to target 1': dist1,
                       'Critic/Q vals heatmap': heatmap_Qs,
                       'Critic/Q next heatmap': heatmap_Q_next,
                       'Critic/Q refs heatmap': heatmap_Qref,
                       'Critic/Q target refs heatmap': heatmap_Qreftarget,
                       'Critic/Bellman terms': bellman_terms,
                       'Critic/Bellman ref terms': bellman_terms_ref,
                       'Critic/Eval critic': eval_crit})

        q_target = rew + (0.97 * q_target).reshape(-1, 1) * (1 - dones)
        q_target = torch.clamp(q_target, min=-100, max=100)

        critic1_loss = F.mse_loss(q1.squeeze(), q_target.squeeze(),
                                  reduction='none')
        critic2_loss = F.mse_loss(q2.squeeze(), q_target.squeeze(),
                                  reduction='none')

        if log_data:            
            err_pass = critic1_loss.detach()
            err_pass = torch.minimum(err_pass, torch.tensor(4).to(self.device))

            heatmap_Rew = self.log_histogram_2d(err_pass.unsqueeze(dim=1), rew,
                                                'Error', 'Reward')
            heatmap_cum = self.log_histogram_2d(err_pass.unsqueeze(dim=1), cum_reward,
                                                'Error', 'Cumulative reward')
            heatmap_Err_Q = self.log_histogram_2d(err_pass.unsqueeze(dim=1), q1,
                                                  'Error', 'Q vals')
            
            wandb.log(
                {'Critic/Target vals clamped': wandb.Histogram(q_target.cpu()),
                 'Critic/Error dist': wandb.Histogram(torch.log(critic1_loss + 1).detach().cpu()),
                 'Critic/Median error': torch.median(critic1_loss).detach().cpu(),
                 'Critic/Max error': critic1_loss.max().detach().cpu(),
                 'Critic error - Reward heatmap': heatmap_Rew,
                 'Critic error - Cumulative reward': heatmap_cum,
                 'Critic error - Q vals': heatmap_Err_Q})

            zero_idx = batch.cum_reward < 0.5
            one_idx = (batch.cum_reward > 0.5) & (batch.cum_reward < 5)
            ten_idx = (batch.cum_reward > 5) & (batch.cum_reward < 20)
            thirty_idx = batch.cum_reward > 20

            wandb.log({'Critic/Zero cum Q vals': wandb.Histogram(q1[zero_idx[:, 0]].detach().cpu()),
                       'Critic/One cum Q vals': wandb.Histogram(q1[one_idx[:, 0]].detach().cpu()),
                       'Critic/Ten cum Q vals': wandb.Histogram(q1[ten_idx[:, 0]].detach().cpu()),
                       'Critic/Thirty cum Q vals': wandb.Histogram(q1[thirty_idx[:, 0]].detach().cpu())})

            rew_thrh = 0.0

            if rew.max() > rew_thrh:
                wandb.log(
                    {'Critic/Nonzero rewards': wandb.Histogram(rew[rew > rew_thrh].detach().cpu())})

        critic1_loss = critic1_loss.mean()
        critic2_loss = critic2_loss.mean()
        
        if log_data:
            wandb.log(
                {'Critic/Critic1 Grad Norm': self.get_gradient(critic1_loss, params, 'Critic1')})

        # if (critic1_loss > 20 or critic2_loss > 20) and self.email:
        #     wandb.alert(title='Critic loss exploded',
        #                 text=f'Critic loss is {critic1_loss}')
        #     self.email = False
        
        z_sample, pdf, mu, std = self.eval_skill_policy(obs, params)

        q_pi_arg = torch.cat([obs, z_sample], dim=1)
        
        q_pi1, q_pi2 = self.eval_critic(q_pi_arg, params)
        q_pi = torch.cat((q_pi1, q_pi2), dim=1)
        q_pi, _ = torch.min(q_pi, dim=1)
        
        if self.use_SAC:
            skill_prior = torch.clamp(pdf.entropy(), max=MAX_SKILL_KL).mean()
        else:
            skill_prior = torch.clamp(kl_divergence(pdf, z_prior), max=MAX_SKILL_KL).mean()
        
        alpha_skill = torch.exp(self.log_alpha_skill).detach()
        skill_prior_loss = alpha_skill * skill_prior
        
        q_val_policy = -torch.mean(q_pi)
        skill_policy_loss = q_val_policy + skill_prior_loss

        policy_losses = [skill_policy_loss]
            
        loss_alpha_skill = torch.exp(self.log_alpha_skill) * \
            (self.delta_skill - skill_prior).detach()

        self.optimizer_alpha_skill.zero_grad()
        loss_alpha_skill.backward()
        self.optimizer_alpha_skill.step()
          
        if log_data:
            with torch.no_grad():
                z_ref, _, mu_ref, _ = self.eval_skill_policy(obs, ref_params)
                q_pi_ref_arg = torch.cat([obs, z_ref], dim=1)
                q_pi_ref, _ = self.eval_critic(q_pi_ref_arg, params)
                mu_diff = F.l1_loss(mu, mu_ref, reduction='none').mean(1)
                mu_max = torch.max(torch.abs(mu), axis=1)[0]
                mu_ref_max = torch.max(torch.abs(mu_ref), axis=1)[0]                

            heatmap_Qpis = self.log_histogram_2d(q_pi.unsqueeze(dim=1), q_pi_ref, 'Q pi', 'Q pi ref')
            heatmap_max = self.log_histogram_2d(mu_max.unsqueeze(dim=1), mu_ref_max.unsqueeze(dim=1),
                                                'Mu max', 'Mu ref max')
            pi_terms = self.log_scatter_3d(q1, q_refs, mu_diff.unsqueeze(dim=1), idxs.unsqueeze(dim=1),
                                           'Q val', 'Q refs', 'Mu diff', 'Idxs')
                
            wandb.log(
                {'Policy/current_q_values': wandb.Histogram(q_pi.detach().cpu()),
                 'Policy/current_q_values_average': q_pi.detach().mean().cpu(),
                 'Policy/current_q_values_max': q_pi.detach().max().cpu(),
                 'Policy/current_q_values_min': q_pi.detach().min().cpu(),
                 'Policy/Z abs value mean': z_sample.abs().mean().detach().cpu(),
                 'Policy/Z std': z_sample.std().detach().cpu(),
                 'Policy/Z distribution': wandb.Histogram(z_sample.detach().cpu()),
                 'Policy/Mean STD': std.mean().detach().cpu(),
                 'Policy/Mu dist': wandb.Histogram(mu.detach().cpu()),
                 'Policy/Q vals': heatmap_Qpis,
                 'Policy/pi terms': pi_terms,
                 'Policy/Max policies': heatmap_max})

            wandb.log(
                {'Priors/Alpha skill': alpha_skill.detach().cpu(),
                 'Priors/skill_prior_loss': skill_prior.detach().cpu()})

            wandb.log(
                {'Critic/Critic loss': critic1_loss,
                 'Critic/Q values': wandb.Histogram(q1.detach().cpu())})

            wandb.log({'Reward Percentage': sum(rew > 0.0) / self.batch_size})
        
        return policy_losses, critic1_loss, critic2_loss

    def eval_skill_prior(self, state, params):
        """Evaluate the policy.

        It takes the current state and params. It evaluates the
        policy.

        Parameters
        ----------
        state : Tensor
            The current observation of agent
        params : dictionary with all parameters for models
            It contains all relevant parameters, e.g., policy, critic,
            etc.
        """
        z_prior = functional_call(self.vae.models['SkillPrior'],
                                  params['SkillPrior'], state)
        return z_prior
    

    def eval_skill_policy(self, state, params):
        sample, pdf, mu, std = functional_call(self.skill_policy,
                                               params['SkillPolicy'],
                                               state)
        return sample, pdf, mu, std

    def eval_critic(self, arg, params, target_critic=False):
        if target_critic:
            name1, name2 = 'Target_critic1', 'Target_critic2'
        else:
            name1, name2 = 'Critic1', 'Critic2'

        q1 = functional_call(self.critic, params[name1], arg)
        q2 = functional_call(self.critic, params[name2], arg)

        return q1, q2

    def computation_state_vae(self, critic_arg, params):
        names = ['StateEncoder', 'StateDecoder']
        z, pdf, mu, std, rec = self.eval_state_vae(critic_arg, params, names)
        rec_loss = -Normal(rec, 1).log_prob(critic_arg).sum(axis=-1)
        N = Normal(0, 1)
        kl_loss = torch.mean(kl_divergence(pdf, N))
        loss = rec_loss.mean() + 0.01 * kl_loss

        with torch.no_grad():
            names2 = ['TargetEncoder', 'TargetDecoder']
            _, _, _, _, rec_target = self.eval_state_vae(critic_arg, params, names2)
            weights = F.mse_loss(critic_arg, rec_target, reduction='none')
            weights = weights.mean(1)
            
        return loss, weights

    def eval_state_vae(self, critic_arg, params, names):
        z, pdf, mu, std = functional_call(self.state_encoder,
                                          params[names[0]],
                                          critic_arg)

        rec = functional_call(self.state_decoder,
                              params[names[1]], z)

        return z, pdf, mu, std, rec

    def log_histogram_2d(self, x, y, xlabel, ylabel):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        data = np.concatenate([x, y], axis=1)
        df = pd.DataFrame(data, columns=[xlabel, ylabel])

        fig_heatmap = px.density_heatmap(df, x=xlabel, y=ylabel,
                                         marginal_x='histogram',
                                         marginal_y='histogram',
                                         nbinsx=60,
                                         nbinsy=60)

        return fig_heatmap

    def log_scatter_3d(self, x, y, z, color, xlabel, ylabel, zlabel, color_label):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        z = z.detach().cpu().numpy()
        color = color.detach().cpu().numpy()

        data = np.concatenate([x, y, z, color], axis=1)
        df = pd.DataFrame(data, columns=[xlabel, ylabel, zlabel, color_label])
        
        fig_scatter = px.scatter_3d(df, x=xlabel, y=ylabel,
                                    z=zlabel, color=color_label)
        fig_scatter.update_layout(scene=dict(aspectmode='cube'))

        return fig_scatter

    def select_policy(self, params, done, obs):
        if done or obs is None:
            self.policy_use = self.policy_use
        else:
            obs = np.array(obs, dtype=np.float32)
            obs = torch.from_numpy(obs).to(self.device)
            obs = obs.reshape(1, -1)
            zs = []
            with torch.no_grad():
                for i in [0, 1]:
                    z, _, _, _ = self.eval_skill_policy(obs, params, i)
                    zs.append(z)
                zs = torch.cat(zs, dim=0)
                obs = obs.repeat(2, 1)
                q_arg = torch.cat([obs, zs], dim=1)
                q_pi1, q_pi2 = self.eval_critic(q_arg, params)
                q_pi = torch.cat((q_pi1, q_pi2), dim=1)
                q_pi, _ = torch.min(q_pi, dim=1)
                self.policy_use = q_pi.argmax().item()
                # item() is used to extract the value and eliminate tensors.

    def distance_to_target_env(self, obs):
        ds = obs[:, -9:]
        d1 = torch.norm(ds[:, 0:3], dim=1).unsqueeze(dim=1)  # Palm to ball
        d2 = torch.norm(ds[:, 3:6], dim=1).unsqueeze(dim=1)  # Palm to target
        d3 = torch.norm(ds[:, 6:9], dim=1).unsqueeze(dim=1)  # Ball to target

        return d1, d2, d3
                                                   

    def get_gradient(self, x, params, key):
        grads = autograd.grad(x, params[key].values(), retain_graph=True,
                              allow_unused=True)

        grads = [grad for grad in grads if grad is not None]
        try:
            grads_vec = nn.utils.parameters_to_vector(grads)
        except RuntimeError:
            pdb.set_trace()
        return torch.norm(grads_vec).detach().cpu()

    def distance_to_params(self, params1, params2, name1, name2):
        with torch.no_grad():
            vec1 = nn.utils.parameters_to_vector(params1[name1].values())
            target_vec1 = nn.utils.parameters_to_vector(params2[name2].values())
        return torch.norm(vec1 - target_vec1)

    def percentile_hist(self, x):
        x = x.detach().cpu().numpy()
        x = np.minimum(x, 2)

        return x              

    def testing(self, params):
        done = False
        obs = None

        rewards = []
        episodes_w_reward = 0
        test_episodes = 500
        length_samples = []
        length_over_time = []

        
        for j in range(test_episodes):
            self.sampler.env.reset()
            check_episode = True
            step = 0
            while not done:
                _, data = self.sampler.skill_iteration(params,
                                                       done,
                                                       obs)

                obs, reward, _, _, l_samp, _, done, _, _ = data

                if step < 64:
                    length_over_time.append(['0-63', l_samp[0]])
                elif step < 128:
                    length_over_time.append(['64-127', l_samp[0]])
                elif step < 196:
                    length_over_time.append(['128-195', l_samp[0]])
                elif step < 257:
                    length_over_time.append(['196-255', l_samp[0]])

                length_samples.append(l_samp[0])
                step += self.level_lengths[l_samp[0]]
                if check_episode and reward > 0.0:
                    episodes_w_reward += 1
                    check_episode = False
                
                rewards.append(reward)

            done = False

        pdb.set_trace()
        print(np.unique(length_samples, return_counts=True))
        evol_lengths = np.array(length_over_time)        
        df = pd.DataFrame(evol_lengths, columns=['Step', 'Length'])
        df.to_csv('runs/case0.csv', index=False)
        average_reward = sum(rewards) / test_episodes
        return average_reward

    def render_results(self, params, foldername):
        test_episodes = 10
        
        for j in range(test_episodes):
            done = False
            obs = None

            frames = []
            self.sampler.env.reset()

            while not done:
                obs, done, frames = self.sampler.skill_iteration_with_frames(params,
                                                                             done=done,
                                                                             obs=obs,
                                                                             frames=frames)

            process_frames(frames, self.env_id, f'{foldername}/test_{j}')
