import random

class Agent:
    def __init__(self, lr, discount, eps_max, eps_min, eps_decrement, num_states, num_actions):
        self.lr = lr
        self.discount = discount
        self.eps = eps_max
        self.eps_min = eps_min
        self.eps_decrement = eps_decrement
        self.num_states = num_states
        self.num_actions = num_actions
        self.Q = dict()

        self.__initialize_q()

    def __initialize_q(self):
        for state in range(self.num_states):
            self.Q[state] = {action: 0.0 for action in range(self.num_actions)}

    def get_best_action(self, state):
        possible_actions = self.Q.get(state)
        best_pair = sorted(list(possible_actions.items()), key=lambda x: x[1], reverse=True)[0]  # x[1] is the score associated to each action
        best_action = best_pair[0]
        best_score = best_pair[1]
        return best_action, best_score

    def get_action(self, state):
        score = 0.0
        if random.random() < self.eps:
            action = random.randint(0, self.num_actions - 1)
        else:
            action, score = self.get_best_action(state)
        return action, score

    def decrement_epsilon(self):
        self.eps = self.eps*self.eps_decrement if self.eps > self.eps_min else self.eps_min
        # IT PREVIOUSLY WAS:
        # self.eps -= 0.0001 if self.eps > self.eps_min else self.eps_min
        # IT WILL NOT LEARN!!! NOT A GOOD WAY TO DECREMENT EPSILON!
        # Agent needs more time to explore randomly.

    def update_q_table(self, state, action, reward, next_state):
        current_q_value = self.Q[state][action]
        next_best_action, next_q_value = self.get_best_action(next_state)
        self.Q[state][action] = current_q_value + self.lr * (reward + self.discount * next_q_value - current_q_value)

        self.decrement_epsilon()