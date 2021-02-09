import gym
import numpy as np
import matplotlib.pyplot as plt
from c2_frozenlake_qtable.agent_sol import Agent

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = Agent(lr=0.01, discount=0.9, eps_max=1.0, eps_min=0.01, eps_dec=0.9999995, num_states=env.observation_space.n,
                  num_actions=env.action_space.n)

    scores = []
    win_pct_list = []
    n_games = 500000

    for episode in range(n_games):
        done = False
        obs = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            score += reward
            obs = next_obs
        scores.append(score)
        if episode % 100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            if episode % 1000 == 0:
                print('episode', episode, 'win pct %.2f' % win_pct, 'epsilon %.2f' % agent.eps)

    plt.plot(win_pct_list)
    plt.show()