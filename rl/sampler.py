"""Create sampler for RL with buffer."""

import sys
sys.path.insert(0, '../')

from utilities.utils import hyper_params, AttrDict
import gym
import d4rl
import numpy as np
from torch.func import functional_call
import torch
import torch.nn.functional as F
import wandb
from scipy import signal

import pdb

WIDTH = 4 * 640
HEIGHT = 4 * 480

class Sampler(hyper_params):
    def __init__(self, skill_policy, decoder, args):
        super().__init__(args)

        self.skill_policy = skill_policy
        self.decoder = decoder
        MAX_EPISODE_STEPS = 257 if self.env_key == 'adroit' else 256

        self.env = gym.make(self.env_id)
        self.env._max_episode_steps = MAX_EPISODE_STEPS

    def skill_execution(self, actions, frames=None):
        obs_trj, rew_trj, done_trj = [], [], []
        aux_frames = []
        
        for i in range(actions.shape[0]):
            next_obs, rew, done, info = self.env.step(actions[i, :])
            if frames is not None:
                if self.env_key != 'kitchen':
                    frame = self.env.sim.render(width=WIDTH, height=HEIGHT,
                                                mode='offscreen',
                                                camera_name='vil_camera')
                    aux_frames.append(frame)
                else:
                    frame = self.env.sim.render(width=WIDTH, height=HEIGHT)
                    aux_frames.append(frame)
                    
            if self.env_key != 'kitchen':
                done = info['goal_achieved'] if len(info) == 1 else True
            obs_trj.append(next_obs)
            rew_trj.append(rew)
            done_trj.append(done)
        if frames is not None:
            frames.append(aux_frames)

        return obs_trj, rew_trj, done_trj, frames

    def skill_step(self, params, obs, policy, frames=None):
        obs_t = torch.from_numpy(obs).to(self.device).to(torch.float32)
        obs_t = obs_t.reshape(1, -1)

        with torch.no_grad():
            z_sample, _, _, _ = functional_call(self.skill_policy,
                                                params[f'SkillPolicy{policy}'],
                                                obs_t)
                
            actions = self.decoder(z_sample, params)
            actions = actions.reshape(-1, actions.shape[-1])

        actions = actions.cpu().detach().numpy()
        clipped_actions = np.clip(actions, -1, 1)
            
        obs_trj, rew_trj, done_trj, frames = self.skill_execution(clipped_actions,
                                                                  frames=frames)

        if frames is not None:
            done = True if sum(done_trj) > 0 else False
            return obs_trj[-1], done, frames

        next_obs_t = torch.from_numpy(obs_trj[-1]).to(self.device).to(torch.float32)
        next_obs_t = next_obs_t.reshape(1, -1)

        with torch.no_grad():
            next_z_sample, _, _, _ = functional_call(self.skill_policy,
                                                     params[f'SkillPolicy{policy}'],
                                                     next_obs_t)
           
        next_obs = obs_trj[-1]
        if self.env_key == 'kitchen':
            rew = sum(rew_trj)
        elif self.env_key == 'adroit':
            rew = rew_trj[-1]

        z = z_sample.cpu().numpy()
        next_z = next_z_sample.cpu().numpy()
        done = True if sum(done_trj) > 0 else False

        return next_obs, rew, z, next_z, done

    def skill_iteration(self, params, done=False, obs=None, policy=0):
        if done or obs is None:
            obs = self.env.reset()

        return obs, self.skill_step(params, obs, policy)

    def skill_iteration_with_frames(self, params, done=False, obs=None, frames=None):
        if done or obs is None:
            obs = self.env.reset()

        frames = self.skill_step(params, obs, frames)

        return frames
    

class NormalReplayBuffer:
    def __init__(self, size, env, lat_dim):
        self.obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.z_buf = np.zeros((size, lat_dim), dtype=np.float32)
        self.next_z_buf = np.zeros((size, lat_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.tracker = np.arange(size)

    def add(self, obs, next_obs, z, next_z, rew, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.z_buf[self.ptr] = z
        self.next_z_buf[self.ptr] = next_z
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        # idxs = np.random.randint(0, self.size, size=batch_size)
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = AttrDict(observations=self.obs_buf[idxs],
                         next_observations=self.next_obs_buf[idxs],
                         z=self.z_buf[idxs],
                         next_z=self.next_z_buf[idxs],
                         rewards=self.rew_buf[idxs],
                         dones=self.done_buf[idxs])
        return batch

    def sample_recent_eps(self, batch_size=128):
        idx = self.size
        batch = AttrDict(observations=self.obs_buf[idx - batch_size:idx],
                         next_observations=self.next_obs_buf[idx - batch_size:idx],
                         z=self.z_buf[idx - batch_size:idx],
                         next_z=self.next_z_buf[idx - batch_size:idx],
                         rewards=self.rew_buf[idx - batch_size:idx],
                         dones=self.done_buf[idx - batch_size:idx],
                         tracker=self.tracker[idx - batch_size:idx])

        return batch


        
class ModifiedReplayBuffer:
    def __init__(self, size, env, lat_dim):
        self.obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.z_buf = np.zeros((size, lat_dim), dtype=np.float32)
        self.next_z_buf = np.zeros((size, lat_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.tracker = np.zeros((size,), dtype=bool)
        self.cum_reward = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.idx_sampler = np.arange(size)
        self.idx_tracker = np.zeros((size, 1), dtype=np.float32)
        self.threshold = 0.0

    def add(self, obs, next_obs, z, next_z, rew, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.z_buf[self.ptr] = z
        self.next_z_buf[self.ptr] = next_z
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.tracker[self.ptr] = True
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        # idxs = np.random.randint(0, self.size, size=batch_size)
        # idxs = np.random.power(1 + .25 * self.ptr / 1e5, size=batch_size)
        # idxs = np.array(idxs * self.size, dtype=np.int32)

        idxs = np.random.randint(0, self.size, size=batch_size)
        
        self.idx_tracker[idxs] += 1

        batch = AttrDict(observations=self.obs_buf[idxs],
                         next_observations=self.next_obs_buf[idxs],
                         z=self.z_buf[idxs],
                         next_z=self.next_z_buf[idxs],
                         rewards=self.rew_buf[idxs],
                         dones=self.done_buf[idxs],
                         cum_reward=self.cum_reward[idxs],
                         idxs=idxs)
        return batch

    def sample_recent_eps(self, batch_size=128):
        idx = self.size
        batch = AttrDict(observations=self.obs_buf[idx - batch_size:idx],
                         next_observations=self.next_obs_buf[idx - batch_size:idx],
                         z=self.z_buf[idx - batch_size:idx],
                         next_z=self.next_z_buf[idx - batch_size:idx],
                         rewards=self.rew_buf[idx - batch_size:idx],
                         dones=self.done_buf[idx - batch_size:idx],
                         tracker=self.tracker[idx - batch_size:idx])

        return batch

    def update_tracking_buffers(self, ep_reward):
        last_ep_idx = np.where(self.done_buf[0:self.ptr - 1])[0].max() + 1
        self.cum_reward[last_ep_idx:self.ptr, :] = ep_reward


    def prune_buffers(self, all_rewards):
        all_rewards = np.array(all_rewards)[-500:]
        self.threshold = np.percentile(all_rewards, 60)
        self.tracker[self.cum_reward < self.threshold] = False
        
        vals = self.tracker.sum()
        
        self.obs_buf[0: vals] = self.obs_buf[self.tracker[:, 0], :]
        self.next_obs_buf[0: vals] = self.next_obs_buf[self.tracker[:, 0], :]
        self.z_buf[0: vals] = self.z_buf[self.tracker[:, 0], :]
        self.next_z_buf[0: vals] = self.next_z_buf[self.tracker[:, 0], :]        
        self.rew_buf[0: vals] = self.rew_buf[self.tracker[:, 0], :]
        self.done_buf[0: vals] = self.done_buf[self.tracker[:, 0], :]
        self.cum_reward[0: vals] = self.cum_reward[self.tracker[:, 0], :]

        self.ptr, self.size = vals, vals
        
        return self.threshold

