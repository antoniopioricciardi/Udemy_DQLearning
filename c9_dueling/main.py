import numpy as np
import argparse
import os
from gym import wrappers

import c9_dueling.agent as Agent
from c9_dueling.env_customization import make_env
from c9_dueling.utils import plot_learning_curve

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Deep Q-Learning: From Paper To Code")
    # '-something' <- optional args
    # 'something' <- non optional args
    parser.add_argument('-env_name', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('-algo', type=str, default='DuelingDDQNAgent')
    parser.add_argument('-plot_name', type=str, default='plot')
    parser.add_argument('-n_games', type=int, default=500)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-gamma', type=float, default=0.99)
    parser.add_argument('-epsilon', type=float, default=1.0, help='Starting value for epsilon')
    parser.add_argument('-epsilon_min', type=float, default=0.1, help='Final value for epsilon')
    parser.add_argument('-epsilon_dec', type=float, default=1e-5, help='Linear factor for decreasing epsilon')
    parser.add_argument('-memory_size', type=int, default=50000, help='Replay buffer memory')
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size for memory sampling')
    parser.add_argument('-replace', type=int, default=1000, help='Steps before replacing target network')
    parser.add_argument('-load_checkpoint', type=bool, default=False, help='If false then the model will train')

    # parse the args
    args = parser.parse_args()
    # access parameters with args.argument

    env_name_path = os.path.join(os.getcwd(), args.env_name)
    agent_path = os.path.join(env_name_path, args.algo)
    models_path = os.path.join(os.getcwd(), 'models')
    plots_path = os.path.join(os.getcwd(), 'plots')
    videos_path = os.path.join(os.getcwd(), 'videos')

    if not os.path.exists(env_name_path):
        os.mkdir(env_name_path)
    if not os.path.exists(models_path):
        os.mkdir(models_path)
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)
    if not os.path.exists(videos_path):
        os.mkdir(videos_path)

    env = make_env(args.env_name)
    best_score = -np.inf  # save model when a new high score is achieved
    load_checkpoint = args.load_checkpoint  # we're training, no need to load checkpoint
    n_games = args.n_games

    agent_ = getattr(Agent, args.algo)
    agent = agent_(input_dims=(env.observation_space.shape), n_actions=env.action_space.n,
                             lr=args.lr, gamma=args.gamma, epsilon=args.epsilon, epsilon_min=args.epsilon_min, epsilon_dec=args.epsilon_dec,
                             memory_size=args.memory_size, batch_size=args.batch_size, replace=args.replace,
                             checkpoint_dir='models/', algo=args.algo, env_name=args.env_name)

    if load_checkpoint:
        agent.load_models()
        videos_path = os.path.join(videos_path, )
        env = wrappers.Monitor(env, videos_path, video_callable=lambda episode_id: True, force=True)  # force overwrites previous video

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
            if load_checkpoint:
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
