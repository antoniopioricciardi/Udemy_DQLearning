import gym
import numpy
import matplotlib.pyplot as plt
import numpy as np

from c4_naive_deep_q_n import agent
from c4_naive_deep_q_n.utils import plot_learning_curve

if __name__=='__main__':
    env = gym.make('CartPole-v1')
    n_episodes = 10000
    scores = []
    eps_history = []
    agent = agent.Agent(n_states=env.observation_space.shape, n_actions=env.action_space.n)

    for episode in range(n_episodes):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            score += reward
            agent.learn(obs, action, next_obs, reward)
            obs = next_obs
        scores.append(score)
        eps_history.append(agent.epsilon)

        if episode % 100 == 0:
            avg_score = np.mean(scores)
            print('episode ', episode, 'score %.1f avg score %.1f epsilon %.2f' % (score, avg_score, agent.epsilon))

    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range(n_episodes)]
    plot_learning_curve(x, scores, eps_history, filename)
