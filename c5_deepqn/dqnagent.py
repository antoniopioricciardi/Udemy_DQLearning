import numpy as np
import torch
from c5_deepqn.deepqn import DeepQN
from c5_deepqn.replaymemory import ReplayMemory


class DQNAgent:
    def __init__(self, input_dims, n_actions, memory_size, batch_size,
                 lr, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_dec=5e-7,
                 replace=1000, algo=None, env_name=None, checkpoint_dir='models/'):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]

        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec

        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.learn_step_counter = 0

        self.memory = ReplayMemory(memory_size, input_dims, n_actions)

        self.eval_DQN = DeepQN(self.lr, self.input_dims, self.n_actions, self.env_name+'_'+self.algo+'_q_eval',
                               self.checkpoint_dir)
        # will be used to compute Q(s',a') - that is for the resulting states
        # We won't perform gradient descent/backprob on this net, only on the eval.
        self.target_DQN = DeepQN(self.lr, self.input_dims, self.n_actions, self.env_name + '_' + self.algo + '_q_target',
                                 self.checkpoint_dir)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # input_dims is a batch, therefore we need to create a batch for every single observation
            state = torch.tensor([state], dtype=torch.float).to(self.eval_DQN.device)
            actions = self.eval_DQN.forward(state)
            action = torch.argmax(actions).item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        # convert each memory element in a pytorch tensor and sent it to the device
        # (lowaercase .tensor() preserves the data type of the underlying numpy array
        states = torch.tensor(state).to(self.eval_DQN.device)
        actions = torch.tensor(action).to(self.eval_DQN.device)
        rewards = torch.tensor(reward).to(self.eval_DQN.device)
        next_states = torch.tensor(next_state).to(self.eval_DQN.device)
        dones = torch.tensor(done, dtype=torch.bool).to(self.eval_DQN.device)

        return states, actions, rewards, next_states, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_DQN.load_state_dict(self.eval_DQN.state_dict())

    def learn(self):
        # if the memory has not been filled yet, don't learn
        if self.memory.mem_counter < self.batch_size:
            return
        self.eval_DQN.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)  # array of numbers in range [0,...,batch_size]
        # a batch of Q(s,a) values - we want to take the value for action for each batch, therefore [indices, actions]
        q_pred = self.eval_DQN.forward(states)[indices, actions]
        # get values for the max action for each next_state
        # dim=1 is the action dimension
        # max returns a named tuple, where 0th element are values, 1st are the indices of max actions.
        # Basically we're saying give me the max for each list of actions.
        q_next = self.target_DQN.forward(next_states).max(dim=1)[0]

        # use dones as a mask
        q_next[dones] = 0.0
        # TARGET VALUE y_j = r_j + gamma*max_a' Q_target(next_state, next_action, previous_theta) if state non terminal
        # r_j otherwise
        # That long Q_target(...) is simply our q_next
        # Therefore using dones as a mask, gamma*q_next is simply 0, and we get just r_j. CLEVER
        q_target = rewards + self.gamma * q_next

        loss = self.eval_DQN.loss(q_target, q_pred).to(self.eval_DQN.device)
        loss.backward()
        self.eval_DQN.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def save_models(self):
        self.eval_DQN.save_checkpoint()
        self.target_DQN.save_checkpoint()

    def load_models(self):
        self.eval_DQN.load_checkpoint()
        self.target_DQN.load_checkpoint()
