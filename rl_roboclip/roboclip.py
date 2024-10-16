import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import gymnasium as gym
import numpy as np

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision.transforms import ToTensor

from experiments.util import Logger
from .s3d.s3dg import S3D


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

def preprocess_video(ep_frames: list[np.array], shorten: bool=True) -> torch.Tensor:
    frames = []
    toTensor = ToTensor()
    for frame in ep_frames:
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        frame = toTensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(frame)

    if shorten:
        frames = frames[::4]
    if len(frames) % 2 != 0:
        frames = frames[1:] # make sure number of frame is even number for s3d input
    return torch.stack(frames).transpose(1, 0).unsqueeze(0)

class RoboClip():
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

        # initialize s3d and demo feature
        s3d_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "s3d"
        s3d_vocab_path = s3d_dir / "s3d_dict.npy"
        s3d_weight_path = s3d_dir / "s3d_howto100m.pth" 
        demo_dir = s3d_dir / "demonstration" / "from_center"

        self.s3d = S3D(s3d_vocab_path, 512)
        self.s3d.load_state_dict(torch.load(s3d_weight_path))
        self.s3d.eval()
        self.s3d = self.s3d.to(args.device)

        frames = []
        clip_names = os.listdir(demo_dir)
        clip_names.sort()
        for clip_name in clip_names:
            frames.append(np.array(Image.open(demo_dir / clip_name)))
        demo_vid = preprocess_video(frames).to(args.device)
        self.demo_emb = self.s3d(demo_vid)["video_embedding"] # [1, 512]

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
            frames.append(env.render()) # collect frames for roboclip reward generation

            # log for updating policy
            actions.append(action.cpu())
            log_prob_actions.append(log_prob_action.cpu())
            values.append(value_pred.cpu())
            rewards.append(0)

            heur_episode_reward += heur_reward

            n_step += 1
        roboclip_reward = 0
        if n_step > 0:
            # normalize episode reward
            heur_episode_reward /= n_step

            # construct roboclip reward
            obs_vid = preprocess_video(frames).to(self.args.device)
            obs_emb = self.s3d(obs_vid)["video_embedding"]
            roboclip_reward = torch.matmul(self.demo_emb, obs_emb.t()).item()
            rewards[-1] = roboclip_reward
        
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

        return total_policy_loss, total_value_loss, heur_episode_reward, roboclip_reward, n_step

    def train(self, env: gym.Env):
        # for each episode
        for _ in tqdm(range(self.args.max_episode)):
            policy_loss, value_loss, episode_reward, roboclip_reward, n_step = self.train_step(env)

            if self.logger is not None:
                self.logger.add(LossPi=policy_loss)
                self.logger.add(LossV=value_loss)
                self.logger.add(EpRet=episode_reward)
                self.logger.add(VLMEpRet=roboclip_reward)
                self.logger.add(EpLen=n_step)

                self.logger.log('LossPi')
                self.logger.log('LossV')
                self.logger.log('EpRet')
                self.logger.log('VLMEpRet')
                self.logger.log('EpLen')
                self.logger.flush()
