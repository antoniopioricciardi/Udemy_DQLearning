import numpy as np
import torch
import random
from torch.nn.utils import clip_grad_norm_

from c5_deepqn.deepqn import DeepQN
from c6_doubledqn.doubledqn import DDQN
from c7_duelingdqn.duelingdqn import DuelingDQN
from atari_dueling.dueling_ddqn import DuelingDDQN
from atari_dueling.ddqn import DDQN
from atari_dueling.replaymemory import ReplayMemory


class Agent:
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
        self.epsilon_new_min = self.epsilon_min*0.1
        self.new_min = False
        self.epsilon_dec = epsilon_dec

        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.learn_step_counter = 0

        self.memory = ReplayMemory(memory_size, input_dims, n_actions)

    def decrement_epsilon(self):
        if self.epsilon <= self.epsilon_min and not self.new_min:
            self.epsilon_min = self.epsilon_min * 0.1
            self.epsilon_dec = self.epsilon_dec * 0.1
            self.new_min = True

        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def choose_action(self, state):
        raise NotImplementedError

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        # convert each memory element in a pytorch tensor and sent it to the device
        # (lowaercase .tensor() preserves the data type of the underlying numpy array
        states = torch.tensor(state).to(self.eval_Q.device)
        actions = torch.tensor(action).to(self.eval_Q.device)
        rewards = torch.tensor(reward).to(self.eval_Q.device)
        next_states = torch.tensor(next_state).to(self.eval_Q.device)
        dones = torch.tensor(done, dtype=torch.bool).to(self.eval_Q.device)

        return states, actions, rewards, next_states, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_Q.load_state_dict(self.eval_Q.state_dict())

    def learn(self):
        raise NotImplementedError

    def save_models(self):
        self.eval_Q.save_checkpoint()
        self.target_Q.save_checkpoint()

    def load_models(self):
        self.eval_Q.load_checkpoint()
        self.target_Q.load_checkpoint()


class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):  # instead of writing class args one by one, we can use *args and **kwargs. The latter is for arg=something case
        super(DQNAgent, self).__init__(*args, **kwargs)
        self.eval_Q = DeepQN(self.lr, self.input_dims, self.n_actions, self.env_name+'_'+self.algo+'_q_eval',
                                       self.checkpoint_dir)
        # will be used to compute Q(s',a') - that is for the resulting states
        # We won't perform gradient descent/backprob on this net, only on the eval.
        self.target_Q = DeepQN(self.lr, self.input_dims, self.n_actions, self.env_name + '_' + self.algo + '_q_target',
                                 self.checkpoint_dir)

    def choose_action(self, state):
        if np.random.random() <= self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # input_dims is a batch, therefore we need to create a batch for every single observation
            state = torch.tensor([state], dtype=torch.float).to(self.eval_Q.device)
            actions = self.eval_Q.forward(state)
            action = torch.argmax(actions).item()
        return action

    def learn(self):
        # if the memory has not been filled yet, don't learn
        if self.memory.mem_counter < self.batch_size:
            return
        self.eval_Q.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)  # array of numbers in range [0,...,batch_size]
        # a batch of Q(s,a) values - we want to take the value for action for each batch, therefore [indices, actions]
        q_pred = self.eval_Q.forward(states)[indices, actions]
        # get values for the max action for each next_state
        # dim=1 is the action dimension

        # max returns a named tuple, where 0th element are values, 1st are the indices of max actions.
        # Basically we're saying give me the max for each list of actions.
        q_next = self.target_Q.forward(next_states).max(dim=1)[0]

        # use dones as a mask
        q_next[dones] = 0.0
        # TARGET VALUE y_j = r_j + gamma*max_a' Q_target(next_state, next_action, previous_theta) if state non terminal
        # r_j otherwise
        # That long Q_target(...) is simply our q_next
        # Therefore using dones as a mask, gamma*q_next is simply 0, and we get just r_j. CLEVER
        q_target = rewards + self.gamma * q_next

        loss = self.eval_Q.loss(q_target, q_pred).to(self.eval_Q.device)
        loss.backward()
        self.eval_Q.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()


class DDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DDQNAgent, self).__init__(*args, **kwargs)

        # self.eval_Q = DDQN(self.lr, self.input_dims, self.n_actions, self.env_name + '_' + self.algo + '_q_eval',
        #                       self.checkpoint_dir)
        # # will be used to compute Q(s',a') - that is for the resulting states
        # # We won't perform gradient descent/backprob on this net, only on the eval.
        # self.target_Q = DDQN(self.lr, self.input_dims, self.n_actions, self.env_name + '_' + self.algo + '_q_target',
        #                         self.checkpoint_dir)

        self.eval_Q = DDQN(self.lr, self.input_dims, self.n_actions,
                                  # self.env_name + '_' + self.algo +
                                  'q_eval',
                                  self.checkpoint_dir)
        # will be used to compute Q(s',a') - that is for the resulting states
        # We won't perform gradient descent/backprob on this net, only on the eval.
        self.target_Q = DDQN(self.lr, self.input_dims, self.n_actions,
                                    # self.env_name + '_' + self.algo +
                                    'q_target',
                                    self.checkpoint_dir)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # input_dims is a batch, therefore we need to create a batch for every single observation
            state = torch.tensor([state], dtype=torch.float).to(self.eval_Q.device)
            actions = self.eval_Q.forward(state)
            action = torch.argmax(actions).item()
        return action

    def learn(self):
        # if memory has not been filled yet, return
        if self.memory.mem_counter < self.batch_size:
            return

        self.eval_Q.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)  # array of numbers in range [0, ..., batch_size]

        q_pred = self.eval_Q.forward(states)[indices, actions]
        q_next = self.target_Q.forward(next_states)
        q_eval = self.eval_Q.forward(next_states)

        next_actions = torch.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        target_next_q_pred = q_next[
            indices, next_actions.detach()]  # detach needed because of a bug in pytorch 1.5.0 which makes that indexing differentiable

        q_target = rewards + self.gamma * target_next_q_pred
        # q_target = rewards + self.gamma * q_next[indices, next_actions]

        loss = self.eval_Q.loss(q_target, q_pred).to(self.eval_Q.device)
        loss.backward()
        # clip_grad_norm_(self.eval_Q.parameters(), 10)
        self.eval_Q.optimizer.step()

        self.learn_step_counter += 1
        self.decrement_epsilon()


class DuelingDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelingDQNAgent, self).__init__(*args, **kwargs)

        self.eval_Q = DuelingDQN(self.lr, self.input_dims, self.n_actions, self.env_name+'_'+self.algo+'_q_eval',
                               self.checkpoint_dir)
        # will be used to compute Q(s',a') - that is for the resulting states
        # We won't perform gradient descent/backprob on this net, only on the eval.
        self.target_Q = DuelingDQN(self.lr, self.input_dims, self.n_actions, self.env_name + '_' + self.algo + '_q_target',
                                 self.checkpoint_dir)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # input_dims is a batch, therefore we need to create a batch for every single observation
            state = torch.tensor([state], dtype=torch.float).to(self.eval_Q.device)
            _, advantages = self.eval_Q.forward(state)
            action = torch.argmax(advantages).item()
        return action

    def learn(self):
        # if the memory has not been filled yet, don't learn
        if self.memory.mem_counter < self.batch_size:
            return
        self.eval_Q.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)  # array of numbers in range [0,...,batch_size]
        # a batch of Q(s,a) values - we want to take the value for action for each batch, therefore [indices, actions]
        value_state, advantages_state = self.eval_Q.forward(states)  # value is a batch
        value_next, advantages_next = self.target_Q.forward(next_states)

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

        loss = self.eval_Q.loss(q_target, q_pred).to(self.eval_Q.device)
        loss.backward()
        self.eval_Q.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()


class DuelingDDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelingDDQNAgent, self).__init__(*args, **kwargs)
        self.eval_Q = DuelingDDQN(self.lr, self.input_dims, self.n_actions,
                                            #self.env_name + '_' + self.algo +
                                  'q_eval',
                                            self.checkpoint_dir)
        # will be used to compute Q(s',a') - that is for the resulting states
        # We won't perform gradient descent/backprob on this net, only on the eval.
        self.target_Q = DuelingDDQN(self.lr, self.input_dims, self.n_actions,
                                              #self.env_name + '_' + self.algo +
                                    'q_target',
                                              self.checkpoint_dir)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # input_dims is a batch, therefore we need to create a batch for every single observation
            state = torch.tensor([state], dtype=torch.float).to(self.eval_Q.device)
            _, advantages = self.eval_Q.forward(state)
            action = torch.argmax(advantages).item()
        return action

    def learn(self):
        # if memory has not been filled yet, return
        if self.memory.mem_counter < self.batch_size:
            return

        self.eval_Q.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()
        indices = np.arange(self.batch_size)  # array of numbers in range [0, ..., batch_size]

        # here we combine the dueling deep q network and double deep q n.
        value_states, adv_states = self.eval_Q.forward(states)
        value_next, adv_next = self.eval_Q.forward(next_states)
        target_val, target_adv = self.target_Q.forward(next_states)

        q_pred = (value_states + (adv_states - torch.mean(adv_states, dim=1, keepdim=True)))[indices, actions]
        q_eval = value_next + (adv_next - torch.mean(adv_next, dim=1, keepdim=True))
        target_pred = (target_val + (target_adv - torch.mean(target_adv, dim=1, keepdim=True)))

        next_actions = torch.argmax(q_eval, dim=1)

        target_pred[dones] = 0

        q_target = rewards + self.gamma * target_pred[indices, next_actions.detach()]

        loss = self.eval_Q.loss(q_pred, q_target).to(self.eval_Q.device)
        loss.backward()
        self.eval_Q.optimizer.step()

        self.learn_step_counter += 1
        self.decrement_epsilon()
