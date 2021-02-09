import numpy as np
import torch
from c6_doubledqn.doubledqn import DDQN
from c6_doubledqn.replaymemory import ReplayMemory


class DDQNAgent:
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

        self.eval_DDQN = DDQN(self.lr, self.input_dims, self.n_actions, self.env_name+'_'+self.algo+'_q_eval',
                               self.checkpoint_dir)
        # will be used to compute Q(s',a') - that is for the resulting states
        # We won't perform gradient descent/backprob on this net, only on the eval.
        self.target_DDQN = DDQN(self.lr, self.input_dims, self.n_actions, self.env_name + '_' + self.algo + '_q_target',
                                 self.checkpoint_dir)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # input_dims is a batch, therefore we need to create a batch for every single observation
            state = torch.tensor([state], dtype=torch.float).to(self.eval_DDQN.device)
            actions = self.eval_DDQN.forward(state)
            action = torch.argmax(actions).item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        # convert each memory element in a pytorch tensor and sent it to the device
        # (lowaercase .tensor() preserves the data type of the underlying numpy array
        states = torch.tensor(state).to(self.eval_DDQN.device)
        actions = torch.tensor(action).to(self.eval_DDQN.device)
        rewards = torch.tensor(reward).to(self.eval_DDQN.device)
        next_states = torch.tensor(next_state).to(self.eval_DDQN.device)
        dones = torch.tensor(done, dtype=torch.bool).to(self.eval_DDQN.device)

        return states, actions, rewards, next_states, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_DDQN.load_state_dict(self.eval_DDQN.state_dict())

    def learn(self):
        # if memory has not been filled yet, return
        if self.memory.mem_counter < self.batch_size:
            return

        self.eval_DDQN.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)  # array of numbers in range [0, ..., batch_size]

        q_pred = self.eval_DDQN.forward(states)[indices, actions]
        q_next = self.target_DDQN.forward(next_states)
        q_eval = self.eval_DDQN.forward(next_states)

        next_actions = torch.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        target_next_q_pred = q_next[indices, next_actions.detach()]  # detach needed because of a bug in pytorch 1.5.0 which makes that indexing differentiable

        q_target = rewards + self.gamma * target_next_q_pred
        # q_target = rewards + self.gamma * q_next[indices, next_actions]

        loss = self.eval_DDQN.loss(q_target, q_pred).to(self.eval_DDQN.device)
        loss.backward()
        self.eval_DDQN.optimizer.step()
        
        self.learn_step_counter += 1
        self.decrement_epsilon()

    def save_models(self):
        self.eval_DDQN.save_checkpoint()
        self.target_DDQN.save_checkpoint()

    def load_models(self):
        self.eval_DDQN.load_checkpoint()
        self.target_DDQN.load_checkpoint()
