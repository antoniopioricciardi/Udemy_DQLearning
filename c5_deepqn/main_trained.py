import numpy as np
import torch
import time
from c5_deepqn.dqnagent import DQNAgent
from c5_deepqn.env_customization import make_env
from c5_deepqn.utils import plot_learning_curve


env = make_env('PongNoFrameskip-v4')

n_games = 10
mem_size = 0
batch_size = 0
agent = DQNAgent(env.observation_space.shape, env.action_space.n, mem_size, batch_size, lr=0, gamma=0, epsilon=0.0,
                 epsilon_min=0.0, epsilon_dec=0.0, replace=1000,
                 algo='DQNAgent', env_name='PongNoFrameskip-v4', checkpoint_dir='models/')

agent.load_models()

for _ in range(n_games):
    done = False
    obs = env.reset()
    while not done:
        action = agent.choose_action(obs)
        obs, reward, done, info = env.step(action)
        time.sleep(0.01)
        env.render()
