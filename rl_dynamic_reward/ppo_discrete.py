import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from torchvision.transforms import ToTensor
from tqdm import tqdm
import gymnasium as gym
import cv2
import clip
from PIL import Image
import imageio.v3 as iio
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from experiments.util import Logger
from rl_baseline.ppo import MLP, ActorCritic, init_weights


def cosine_similarity(vec1: np.array, vec2: np.array):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class DynamicReward:
    """
    Novel reward function used.
        1. extract CLIP features from every demo 
        2. calculate dtw of current trajectory with the most similar one
    """
    def __init__(self, args):
        self.logger = None
        self.args = args

        # initialize policy
        self.model = ActorCritic(
            actor = MLP(args.obs_dim, args.hidden_dim, args.act_dim),
            critic = MLP(args.obs_dim, args.hidden_dim, 1)
        )
        self.model.apply(init_weights)
        self.model = self.model.to(args.device)

        # load CLiP
        self.clip, self.preprocess = clip.load("ViT-B/32", device=self.args.device)
        
        # extract features of demo videos in advance
        self.demo_feats = []
        
        demo_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "demonstration"
        demo_names = os.listdir(demo_dir)
        for demo_name in demo_names:
            demo_frames = os.listdir(demo_dir / demo_name)
            demo_frames.sort()
            
            frames = []
            for idx in demo_frames:
                frames.append(Image.open(demo_dir / demo_name / idx))
            self.demo_feats.append([self.extract_feat(frame) for frame in frames])

        # initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def extract_feat(self, frame: Image.Image):
        with torch.no_grad():
            image = self.preprocess(frame).unsqueeze(0).to(self.args.device)
            return self.clip.encode_image(image).cpu().numpy().squeeze(0)
        
    def construct_clip_reward(self, ep_frames: List[np.array], window_size: int = 32) -> List:
        """
        Compute dtw distance between current episode video and most similar demo
        based on clip features
        """
        rew = []
        ep_feats = [self.extract_feat(Image.fromarray(frame)) for frame in ep_frames]
        
        window_size = min(window_size, len(ep_frames))
        for start_idx in range(len(ep_frames)-window_size+1):
            # find the most similar demo
            frame_sim = [cosine_similarity(ep_feats[start_idx], feats[0]) for feats in self.demo_feats]
            idx = np.argmax(frame_sim)
            
            # calculate distance
            dtw_dist, _ = fastdtw(ep_feats[start_idx:start_idx+window_size], self.demo_feats[idx], dist=euclidean)
            rew.append(1.0 / (1.0 + dtw_dist))

        return rew
        
    def set_logger(self, log_dir, logger: Logger):
        self.log_dir = log_dir
        self.logger = logger
    
    def train_step(self, env):
        states = []
        actions = []
        log_prob_actions = []
        values = []
        frames = [] # list of frames(ndarray shape:[400, 600, 3])
        heur_episode_reward = 0
        term = False
        trunc = False

        self.model.train()
        state, _ = env.reset()

        n_step = 0
        while not (term or trunc):
            state = torch.FloatTensor(state).unsqueeze(0)
            states.append(state)

            action_pred, value_pred = self.model(state.to(self.args.device))

            # sample action
            action_prob = F.softmax(action_pred, dim = -1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)

            state, heur_reward, term, trunc, _ = env.step(action.item())
            frames.append(env.render()) # collect frames for dynamic reward generation

            # log for updating policy
            actions.append(action.cpu())
            log_prob_actions.append(log_prob_action.cpu())
            values.append(value_pred.cpu())

            heur_episode_reward += heur_reward

            n_step += 1
            
        dynamic_reward = 0
        if n_step > 0:
            # normalize episode reward
            heur_episode_reward /= n_step

            # construct dynamic reward
            rewards = self.construct_clip_reward(frames)
            rewards.extend([0] * (len(values) - len(rewards) - 1))
            rewards.append(values[-1])
            dynamic_reward += sum(rewards)
        
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_prob_actions = torch.cat(log_prob_actions)
        values = torch.cat(values).squeeze(-1)

        # calculate return
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + R * self.args.discount_factor
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / returns.std() # normalize

        # calculate advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / advantages.std()

        # PPO update
        total_policy_loss = 0
        total_value_loss = 0

        states = states.to(self.args.device).detach()
        actions = actions.to(self.args.device).detach()
        log_prob_actions = log_prob_actions.to(self.args.device).detach()
        advantages = advantages.to(self.args.device).detach()
        returns = returns.to(self.args.device).detach()

        for _ in range(self.args.ppo_steps):
            action_pred, value_pred = self.model(states)
            value_pred = value_pred.squeeze(-1)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)

            # calculate policy ratio
            new_log_prob_actions = dist.log_prob(actions)
            policy_ratio = (new_log_prob_actions - log_prob_actions).exp()

            # calculate policy loss
            unclamped_adv = policy_ratio * advantages
            clamped_adv = torch.clamp(policy_ratio, min=1.0-self.args.ppo_clip, max=1.0+self.args.ppo_clip) * advantages
            policy_loss = -torch.min(unclamped_adv, clamped_adv).mean()

            # calculate value loss
            value_loss = F.smooth_l1_loss(returns, value_pred).mean()

            # take step
            self.optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        total_policy_loss /= self.args.ppo_steps
        total_value_loss /= self.args.ppo_steps

        return total_policy_loss, total_value_loss, heur_episode_reward, dynamic_reward.item(), n_step

    def train(self, env: gym.Env):
        max_episode_reward = float("-INF")

        # for each episode
        for episode in tqdm(range(self.args.max_episode)):
            policy_loss, value_loss, episode_reward, dynamic_reward, n_step = self.train_step(env)

            if self.logger is not None:
                self.logger.add(LossPi=policy_loss)
                self.logger.add(LossV=value_loss)
                self.logger.add(EpRet=episode_reward)
                self.logger.add(VLMEpRet=dynamic_reward)
                self.logger.add(EpLen=n_step)

                self.logger.log('LossPi')
                self.logger.log('LossV')
                self.logger.log('EpRet')
                self.logger.log('VLMEpRet')
                self.logger.log('EpLen')
                self.logger.flush()

                # save checkpoint
                if episode_reward > max_episode_reward:
                    if episode > 0:
                        os.remove(self.log_dir / f"{max_episode_reward}.ckpt")
                    torch.save(self.model.state_dict(), self.log_dir / f"{episode_reward}.ckpt")
                    max_episode_reward = episode_reward
