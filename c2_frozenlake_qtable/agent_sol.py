import random
import numpy as np

class Agent:
    def __init__(self, lr, discount, eps_max, eps_min, eps_dec, num_states, num_actions):
        self.lr = lr
        self.discount = discount
        self.eps = eps_max
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.num_states = num_states
        self.num_actions = num_actions
        self.Q = dict()

        self.__initialize_q()

    def __initialize_q(self):
        for state in range(self.num_states):
            for action in range(self.num_actions):
                self.Q[(state, action)] = 0.0

    def choose_action(self, state):
        if np.random.random() < self.eps:
            action = np.random.choice([i for i in range(self.num_actions)])
        else:
            actions = np.array([self.Q[(state, a)] for a in range(self.num_actions)])
            '''NOTE: if there's a tie, np.argmax returns always the lowest index element.
            One could write a custom function to change that, maybe to select at random amongst ties.'''
            action = np.argmax(actions)
        return action

    def decrement_epsilon(self):
        self.eps = self.eps*self.eps_dec if self.eps > self.eps_min else self.eps_min

    def learn(self, state, action, reward, next_state):
        actions = np.array([self.Q[(next_state, a)] for a in range(self.num_actions)])
        a_max = np.argmax(actions)

        # update q-table
        self.Q[(state, action)] += self.lr * (reward + self.discount*self.Q[(next_state, a_max)] - self.Q[(state, action)])

        #decrement epsilon
        self.decrement_epsilon()