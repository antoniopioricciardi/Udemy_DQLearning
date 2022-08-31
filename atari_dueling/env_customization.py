import gym
import numpy as np
from collections import deque
import collections
import cv2


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat_n=4, clip_rewards=False, no_ops=0, fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.env = env
        self.shape = env.observation_space.shape
        # we only save two frames independently of repeat_n because we are repeating frames and
        # we do not care what happens in those frames. We only need last frame (or last -1) because that
        # will be returned as a new state
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.repeat_n = repeat_n
        self.clip_rewards = clip_rewards
        # no_ops and fire_first are used to compare scores to the paper
        self.no_ops = no_ops  # number of random operations to perform when agent resets (so that agent starts in a different state every time)
        self.fire_first = fire_first  # some envs need o fire first in order to begin playing
        self.lives = 0

    def step(self, action):
        """
        Overloading default env step function, so that we can get maximum value among frames and
        repeat the same action many times.
        We take the maximum among frames because the atari env sometimes "flashes" between frames, meaning
        that some things may not be displayed. Therefore if among frames something is not black (the background color)
        we want to take it.
        :param action:
        :return:
        """
        tot_reward = 0.0
        done = False  # this done is needed because we want to return it (as a check whether env is done or not
        for i in range(self.repeat_n):
            obs, reward, done, info = self.env.step(action)
            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives:
                reward = -1
            if self.clip_rewards:
                reward = np.clip(np.array([reward]), -1, 1)[0]  # clip the reward in -1, 1, then take first element (we need the scalar, not an array)
            tot_reward += reward
            index = i % 2
            self.frame_buffer[index] = obs
            self.lives = lives
            if done:
                break
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])  # element-wise comparison
        return max_frame, tot_reward, done, info

    def reset(self):
        """Overloading default env reset function"""
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for i in range(no_ops):
            _, _, done, _ = self.env.step(0)  # could put a random acton here
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)  # perform the FIRE action
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = obs
        # self.env.unwrapped.ale.lives()
        return obs


'''https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py'''
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
               Done by DeepMind for the DQN and co. since it helps value estimation.
               """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.env = env
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=self.shape, dtype=np.float32)  # set the obs space

    '''overload observation function'''
    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)  # convert to grayscale
        observation = cv2.resize(observation, self.shape[1:],
                                 interpolation=cv2.INTER_AREA)  # resize the obs (INTER_AREA is best for shrinking)
        observation = np.array(observation, dtype=np.uint8).reshape(self.shape)  # convert the obs to numpy array and reshape it to the new shape
        observation = observation / 255.0
        return observation


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, stack_size):
        super(StackFrames, self).__init__(env)
        self.env = env
        # obs_space states go from a min of 0.0 to a max of 1.0. They are repeated because we want to stack frames
        # and give this stack of frames as input to our NN
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(stack_size, axis=0),
            env.observation_space.high.repeat(stack_size, axis=0),
            dtype=np.float32)
        self.stack_size = stack_size
        self.stack = deque(maxlen=stack_size)

    def reset(self):
        self.stack.clear()  # clear stack
        obs = self.env.reset()  # reset env
        for i in range(self.stack_size):  # append obs to stack stack_size times
            self.stack.append(obs)
        # convert stack to numpy array, reshape and return it
        return np.array(self.stack).reshape(self.observation_space.low.shape)  # just give a shape, low or high it's the same

    '''overload observation function'''
    def observation(self, observation):
        # observation is a list of stack_size (true) observations
        # append obs to the stack
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)


def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False, no_ops=0, episodic_life=False, fire_first=False,
             render_mode="rgb_array"):
    """

    :param env_name:
    :param shape:
    :param repeat: repeat frequency - frames to repeat and to stack
    :param clip_rewards: used during testing
    :param no_ops: used during testing
    :param episodic_life: whether the loss of a life should be considered as an episode
    :param fire_first: used during testing
    :return:
    """
    # operations above basically stacks the changes on the environment
    env = gym.make(env_name, render_mode=render_mode)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env
