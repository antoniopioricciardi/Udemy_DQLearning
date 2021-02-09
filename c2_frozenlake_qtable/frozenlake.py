import numpy as np
import matplotlib.pyplot as plt
from frozenlake_qtable.agent import Agent
import gym

if __name__ == '__main__':

    lr = 0.001
    gamma = 0.9
    eps_max = 1.0
    eps_min = 0.01
    eps_decrement = 0.9999995
    num_episodes = 500000


    env = gym.make('FrozenLake-v0')
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    agent = Agent(lr, gamma, eps_max, eps_min, eps_decrement, num_states, num_actions)
    scores = []
    avg_scores = []

    for episode in range(num_episodes):
        done = False
        score = 0
        obs = env.reset()  # observations, NOT states (although it doesn't really matter)
        while not done:
            action, value = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.update_q_table(obs, action, reward, next_obs)
            score += reward
            obs = next_obs
        scores.append(score)

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            if episode % 1000 == 0:
                print('episode', episode, 'win pct %.2f' % avg_score, 'epsilon %.2f' % agent.eps)

    plt.plot(avg_scores)
    plt.show()

    '''IT WORKS! My only mistake was in the way I decremented epsilon.
    I just performed epsilon -= 0.001 or 0.0001 if epsilon > eps_min. This was tooooo fast!
    epsilon*epsilon_dec is a better way, not only when the random action is taken, but at every learn step.
    If we don't decrease at each learning step, then it will destabilize training because
    at later episodes we will still be taking random actions with a certain probability.'''
