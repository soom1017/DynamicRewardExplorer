from typing import Optional

import re
import io
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import gymnasium as gym
import numpy as np
import openai
import base64

from PIL import Image
from tqdm import tqdm
from pathlib import Path

from experiments.util import Logger


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        
        return action_pred, value_pred

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


class OpenaiRewardGenerator():
    def __init__(self):
        openai_key_path = r"C:\Users\litco\Desktop\project\openai_key.txt"
        with open(openai_key_path, "r") as f:
            self.openai_key = f.readline().rstrip()
    
    def preprocess_frame(self, frame: np.array):
        frame = Image.fromarray(frame)

        with io.BytesIO() as output:
            frame.save(output, format='PNG')
            binary = output.getvalue()

        return base64.b64encode(binary).decode('utf-8')

    def generate(self, frame: np.array) -> Optional[float]:
        client = openai.OpenAI(api_key=self.openai_key)        

        system_prompt = """
        You are a vision based rl reward generator.
        User will not provide any coordinates.
        Generate reward based on image analysis only.
        ONLY return the floating point reward number
        """        

        user_prompt = """
        This is a picture of a purple lunarlander landing.
        The goal is to land in between yellow flags.
        Accounting that the lunarlander can be placed wherever in the black backgroud,
        generate a reward from 0 to 1 (floating point number up to 3 decimal points) to train reinforcement learning model.
        ONLY return the floating point reward number.
        """        

        try:
            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":  f"data:image/jpeg;base64,{self.preprocess_frame(frame)}"
                            },
                        },
                    ],
                }
            ])
        except Exception as e:
            print("[openai] couldn't retrieve generated result")
            return None

        # extract reward from response
        content = response.choices[0].message.content
        match = re.search(r'[-+]?\d*\.\d+|\d+', content)
        if match:
            return float(match.group())
        else:
            print(f"[openai] returned non floating point number: {content}")
            return None


class GPT():
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

        self.reward_generator = OpenaiRewardGenerator()

        # initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def set_logger(self, log_dir, logger: Logger):
        self.log_dir = log_dir
        self.logger = logger
    
    def train_step(self, env):
        states = []
        actions = []
        log_prob_actions = []
        values = []
        rewards = []
        vlm_episode_reward = 0
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
            vlm_reward = self.reward_generator.generate(frame=env.render())
            if vlm_reward is None:
                vlm_reward = 0

            # log for updating policy
            actions.append(action.cpu())
            log_prob_actions.append(log_prob_action.cpu())
            values.append(value_pred.cpu())
            rewards.append(vlm_reward)

            heur_episode_reward += heur_reward
            vlm_episode_reward += vlm_reward

            n_step += 1

        if n_step > 0:
            # normalize episode reward
            heur_episode_reward /= n_step
            vlm_episode_reward /= n_step
        
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

        return total_policy_loss, total_value_loss, heur_episode_reward, vlm_episode_reward, n_step

    def train(self, env: gym.Env):
        max_episode_reward = float("-INF")

        # for each episode
        for episode in tqdm(range(self.args.max_episode)):
            policy_loss, value_loss, episode_reward, vlm_episode_reward, n_step = self.train_step(env)

            if self.logger is not None:
                self.logger.add(LossPi=policy_loss)
                self.logger.add(LossV=value_loss)
                self.logger.add(EpRet=episode_reward)
                self.logger.add(VLMEpRet=vlm_episode_reward)
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
