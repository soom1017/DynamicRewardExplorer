from typing import Tuple, Optional
import os

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.distributions.normal import Normal

import gymnasium as gym
from tqdm import tqdm

from experiments.util import Logger

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
        
def log_prob_from_dist(dist: Normal, act: Tensor) -> Tensor:
    return dist.log_prob(act).sum(axis=-1)
        
# Actor-Critic
class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, act_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )
        self.log_std = torch.nn.Parameter(torch.as_tensor(
            -0.5 * np.ones(act_dim, dtype=np.float32)
        ))

    def get_distribution(self, obs: Tensor) -> Normal:
        mu = self.mu(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def forward(self, obs: Tensor, act: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        dist = self.get_distribution(obs)
        log_prob = None
        if act is not None:
            log_prob = log_prob_from_dist(dist, act)
        return dist, log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: Tensor) -> Tensor:
        return torch.squeeze(self.critic(obs), -1)

class ActorCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pi = Actor(args.obs_dim, args.hidden_dim, args.act_dim, args.dropout)
        self.v = Critic(args.obs_dim, args.hidden_dim, args.dropout)
        
    def step(self, obs: Tensor):
        with torch.no_grad():
            dist = self.pi.get_distribution(obs)
            act = dist.sample()
        return act.squeeze().cpu().numpy()
        
    def infer(self, obs: Tensor):
        with torch.no_grad():
            dist = self.pi.get_distribution(obs)
        return dist.loc
        
class A2CAgent:
    def __init__(self, args, train = True, ckpt_path = None):
        self.model = ActorCritic(args)
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))
        self.model = self.model.to(args.device)
        self.max_episode = args.max_episode
        self.gamma = args.gamma
        self.device = args.device
        self.logger = None
        
        if train:
            self.model.apply(init_weights)
            self.pi_optim = Adam(self.model.pi.parameters(), lr=args.lr)
            self.v_optim = Adam(self.model.v.parameters(), lr=args.lr)
            
    def set_logger(self, log_dir, logger: Logger):
        self.log_dir = log_dir
        self.logger = logger

    def train(self, env: gym.Env):
        max_epRet = float("-INF")

        for episode in tqdm(range(self.max_episode)):
            state, info = env.reset()
            num_step = 0
            ep_ret = 0
            term = False
            trunc = False

            while not (term or trunc):
                num_step += 1

                obs = torch.tensor([state], dtype=torch.float).to(self.device)
                
                action = self.model.step(obs)
                next_state, reward, term, trunc, _ = env.step(action)
                pi_loss, v_loss = self.train_step(state, action, reward, next_state, term | trunc)
                
                if self.logger is not None:
                    self.logger.add(LossPi=pi_loss)
                    self.logger.add(LossV=v_loss)

                state = next_state
                ep_ret += reward
                
            ep_ret = ep_ret / num_step if num_step != 0 else 0
            
            if self.logger is not None:
                # log
                self.logger.add(EpLen=num_step)
                self.logger.add(EpRet=ep_ret)
                
                self.logger.log('EpLen')
                self.logger.log('EpRet')
                self.logger.log('LossPi')
                self.logger.log('LossV')
                self.logger.log('Entropy')
                self.logger.flush()
                # save checkpoint
                if ep_ret > max_epRet:
                    if episode > 0:
                        os.remove(self.log_dir / f"{max_epRet}.ckpt")
                    torch.save(self.model.state_dict(), self.log_dir / f"{ep_ret}.ckpt")
                    max_epRet = ep_ret

    def train_step(self, obs, act, rew, next_obs, done):
        obs = torch.tensor([obs], dtype=torch.float).to(self.device)
        next_obs = torch.tensor([next_obs], dtype=torch.float).to(self.device)
        act = torch.tensor([act], dtype=torch.float).to(self.device)
        rew = torch.tensor([rew], dtype=torch.float).to(self.device)
        done = torch.tensor([done], dtype=torch.float).to(self.device)

        # 1-step actor critic
        curr_Q = self.model.v(obs)
        next_Q = self.model.v(next_obs)
        expected_Q = rew + self.gamma * next_Q * (1 - done)
        TD = expected_Q - curr_Q    # TD error: r_t + gamma * Q(s_t+1) - Q(s_t)

        v_loss = nn.MSELoss()(curr_Q, expected_Q.detach())
        self.v_optim.zero_grad()
        v_loss.backward()
        self.v_optim.step()

        dist, log_prob = self.model.pi(obs, act)
        pi_loss = -(log_prob * TD.detach()).mean()
        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()
        
        # extra useful information
        if self.logger is not None:
            ent = dist.entropy().mean().item()
            self.logger.add(Entropy=ent)

        return pi_loss.item(), v_loss.item()
