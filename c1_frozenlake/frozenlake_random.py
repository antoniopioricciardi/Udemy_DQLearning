import gym
import numpy as np
import matplotlib.pyplot as plt
'''
Run FrozenLake with action random sampling (take a random action at each step).
Then plot, every 10 games, the average score of the last 10 games.
As we can see, the average score in 10 games is usually about 0.01 or 0.02 at most,
meaning that it doesn't succeed more than 2 times in 10 games.
'''

env = gym.make('FrozenLake-v0')
scores = []
avg_scores = []
epochs = 1000
for epoch in range(epochs):
    done = False
    score = 0
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        #print(action)
        state, reward, done, _ = env.step(action)
        env.render()
        print(state)
        score += reward
    scores.append(score)

    # keep track of avg win %, every 10 games
    if epoch % 10 == 0:
        average = np.mean(scores[-10:])
        avg_scores.append(average)
        print('Epoch:', epoch, ' - success % = ', average)
print('Epoch:', epoch, ' - success % = ', average)

plt.plot(avg_scores)
plt.show()