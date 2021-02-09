import numpy as np
import torch
from c7_duelingdqn.duelingdqn import DuelingDQN
from c7_duelingdqn.replaymemory import ReplayMemory


class DuelingDQNAgent:
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

        self.eval_DuelingDQN = DuelingDQN(self.lr, self.input_dims, self.n_actions, self.env_name+'_'+self.algo+'_q_eval',
                               self.checkpoint_dir)
        # will be used to compute Q(s',a') - that is for the resulting states
        # We won't perform gradient descent/backprob on this net, only on the eval.
        self.target_DuelingDQN = DuelingDQN(self.lr, self.input_dims, self.n_actions, self.env_name + '_' + self.algo + '_q_target',
                                 self.checkpoint_dir)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # input_dims is a batch, therefore we need to create a batch for every single observation
            state = torch.tensor([state], dtype=torch.float).to(self.eval_DuelingDQN.device)
            _, advantages = self.eval_DuelingDQN.forward(state)
            action = torch.argmax(advantages).item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        # convert each memory element in a pytorch tensor and sent it to the device
        # (lowaercase .tensor() preserves the data type of the underlying numpy array
        states = torch.tensor(state).to(self.eval_DuelingDQN.device)
        actions = torch.tensor(action).to(self.eval_DuelingDQN.device)
        rewards = torch.tensor(reward).to(self.eval_DuelingDQN.device)
        next_states = torch.tensor(next_state).to(self.eval_DuelingDQN.device)
        dones = torch.tensor(done, dtype=torch.bool).to(self.eval_DuelingDQN.device)

        return states, actions, rewards, next_states, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_DuelingDQN.load_state_dict(self.eval_DuelingDQN.state_dict())

    def learn(self):
        # if the memory has not been filled yet, don't learn
        if self.memory.mem_counter < self.batch_size:
            return
        self.eval_DuelingDQN.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)  # array of numbers in range [0,...,batch_size]
        # a batch of Q(s,a) values - we want to take the value for action for each batch, therefore [indices, actions]
        value_state, advantages_state = self.eval_DuelingDQN.forward(states)  # value is a batch
        value_next, advantages_next = self.target_DuelingDQN.forward(next_states)

        # torch.add(a,b) should be the same as a+b as long as a,b are tensors
        q_pred = (value_state + (advantages_state - torch.mean(advantages_state, dim=1, keepdim=True)))[indices, actions]
        q_next = (value_next + (advantages_next - torch.mean(advantages_next, dim=1, keepdim=True))).max(dim=1)[0]

        # q_pred = torch.add(value_state,
        #                (advantages_state - advantages_state.mean(dim=1, keepdim=True)))[indices, actions]
        # q_next = torch.add(value_next,
        #               (advantages_next - advantages_next.mean(dim=1, keepdim=True))).max(dim=1)[0]

        # use dones as a mask
        q_next[dones] = 0.0
        # TARGET VALUE y_j = r_j + gamma*max_a' Q_target(next_state, next_action, previous_theta) if state non terminal
        # r_j otherwise
        # That long Q_target(...) is simply our q_next
        # Therefore using dones as a mask, gamma*q_next is simply 0, and we get just r_j. CLEVER
        q_target = rewards + self.gamma * q_next

        loss = self.eval_DuelingDQN.loss(q_target, q_pred).to(self.eval_DuelingDQN.device)
        loss.backward()
        self.eval_DuelingDQN.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def save_models(self):
        self.eval_DuelingDQN.save_checkpoint()
        self.target_DuelingDQN.save_checkpoint()

    def load_models(self):
        self.eval_DuelingDQN.load_checkpoint()
        self.target_DuelingDQN.load_checkpoint()
