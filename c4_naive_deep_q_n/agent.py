from c4_naive_deep_q_n import network
import numpy as np
import torch

'''DO NOT USE lr in the Q update function, because the optimizer will take care of it.'''
class Agent:
    def __init__(self, n_states, n_actions, lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_dec=1e-5):
        self.Q = network.LinearDeepQNetwork(lr, n_states, n_actions)
        self.n_actions = n_actions
        self.n_states = n_states
        self.action_space = [i for i in range(n_actions)]

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            # choice is more efficient than randint
            action = np.random.choice(self.action_space)
        else:
            # action = np.argmax(np.array(self.Q[(state, a)] for a in range(self.n_actions)))
            state = torch.tensor(state, dtype=torch.float).to(self.Q.device)  # convert state to a tensor and send it to device
            actions = self.Q.forward(state)
            action = torch.argmax(actions).item()  #.item() is needed to get the actual value (a numpy array)
        return action


    def learn(self, state, action, next_state, reward):
        self.Q.optimizer.zero_grad()

        states = torch.tensor(state, dtype=torch.float).to(self.Q.device)
        actions = torch.tensor(action).to(self.Q.device)
        next_states = torch.tensor(next_state, dtype=torch.float).to(self.Q.device)
        rewards = torch.tensor(reward).to(self.Q.device)

        q_pred = self.Q.forward(states)[actions]  # get the score for the taken action

        next_q_pred = self.Q.forward(next_states).max()  # returns the maximum score

        # compute the target (direction we want to move in)
        # this is: [gamma*max_a(Q(next_s, next_a))]
        q_target = reward + self.gamma*next_q_pred  # rewards ?
        # this is q_target - Q(s,a)
        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        # no need to do: Q(s,a) + lr*loss, optimizer.step() does it for us
        self.Q.optimizer.step()
        self.decrement_epsilon()