import numpy as np
import torch
from gym import wrappers
from c5_deepqn.dqnagent import DQNAgent
from c5_deepqn.env_customization import make_env
from c5_deepqn.utils import plot_learning_curve

if __name__=='__main__':
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf  # save model when a new high score is achieved
    load_checkpoint = True  # we're training, no need to load checkpoint
    n_games = 500
    # epsilon will get to 0.1 in around 100000 steps
    agent = DQNAgent(input_dims=(env.observation_space.shape), n_actions=env.action_space.n,
                     lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_dec=1e-5,
                     memory_size=50000, batch_size=32, replace=1000,
                     checkpoint_dir='models/', algo='DQNAgent', env_name='PongNoFrameskip-v4')

    env = wrappers.Monitor(env, 'videos/', video_callable=lambda episode_id: True,
                           force=True)  #  force overwrites previous video

    if load_checkpoint:
        agent.load_models()

    # for saving plot
    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    # steps array is for plotting scores wrt steps, instead of games played
    # because games are highly variable, can be short or long games. Steps are steps.
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            env.render()

            score += reward
            if not load_checkpoint:
                agent.store_transition(obs, action, reward, next_obs, int(done))
                agent.learn()
            obs = next_obs
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score: ', score,
              'average score %.1f best score %.1f epsilon %.2f' % (avg_score, best_score, agent.epsilon),
              'steps ', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score
        eps_history.append(agent.epsilon)

    plot_learning_curve(steps_array, scores, eps_history, figure_file)
