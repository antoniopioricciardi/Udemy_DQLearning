import gym
import numpy as np
import matplotlib.pyplot as plt

'''TRYING A (very stupid) DETERMINISTIC VERSION'''
'''
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
0 = LEFT, 1 = DOWN, 2 = RIGHT, 3 = UP
'''
env = gym.make('FrozenLake-v0')

# let's map states to actions. First state is 0, then, 1, 2, 3, second row starts at 4, and so on
policy = {0: 2, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 2, 10: 1, 13: 2, 14: 2}

scores = []
win_pct = []
num_epochs = 1000
for epoch in range(num_epochs):
    done = False
    state = env.reset()
    score = 0
    while not done:
        action = policy[state]
        state, reward, done, info = env.step(action)
        score += reward
        env.render()
    scores.append(score)
    if epoch % 10 == 0:
        avg = np.mean(scores[-10:])
        win_pct.append(avg)
plt.plot(win_pct)
plt.show()
