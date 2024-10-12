import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
import wandb

import numpy as np
import torch
import gymnasium as gym

from .util import Logger
from rl_baseline.a2c import A2CAgent as A2CBaseline

cur_dir = Path(os.path.dirname(__file__))

AGENT = {
    'a2c_baseline': A2CBaseline,
}

def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    state, info = env.reset(seed=seed)

    args.train.obs_dim = env.observation_space.shape[0]
    args.train.act_dim = env.action_space.shape[0]
    args.train.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.log = args.get('log', True)
    
    agent = AGENT[args.agent](args.train)
    
    if args.log:
        # init wandb logger
        wandb.init(project="LunarLander", name=args.agent, config=dict(args), entity="flybyml")
        wandb.watch(agent.model)
        # configure model checkpoint save and log dir
        log_dir = Path(cur_dir / "log" / args.agent)
        os.makedirs(log_dir, exist_ok=True)
        
        agent.set_logger(log_dir = log_dir, logger = Logger())
    
    agent.train(env)
    
if __name__ == "__main__":
    config_name = sys.argv[1]
    conf = OmegaConf.load(cur_dir / "config" / f"{config_name}.yaml")
    conf.merge_with_cli()
    
    main(conf)