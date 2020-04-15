'''

Implementation of Q-learning with linear function apporximation
to solve the mountain car environment.

Author: BockSong

'''

import sys
import math
import numpy as np
from environment import MountainCar

Debug = True

action_space = [0, 1, 2]

class qlearning(object):
    def __init__(self, mode, epsilon, gamma, learning_rate):
        self.mode = mode
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = learning_rate

        self.env = MountainCar(mode)
        self.env.reset()

        self.state = 0 # TODO: in "raw" or "tile" representation
        self.s = 1
        self.a = len(action_space)
        self.W = np.zeros((self.s, self.a)) # TODO: dimention?
        self.b = 0
        self.q = np.zeros((self.s, self.a))

    def linear_approx(self, state, action):
        return np.dot(state.T, self.W) + self.b

    # choose an action based on epsilon-greedy method
    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(action_space) + 1)
        else:
            # In case of multiple maximum values, the indice of the first occurrence is returned.
            return np.argmax(self.q[self.state])

    def run(self, weight_out, returns_out, episodes, max_iterations):
        with open(returns_out, 'w') as f_returns:
            # perform training
            for episode in range(episodes):
                rewards = 0
                for i in range(max_iterations):
                    # run step
                    action = self.select_action()
                    self.state, reward, done = self.env.step(action)

                    # update parameters
                    self.W = self.W - self.lr * (self.q - (reward + self.gamma * np.max()))

                    if done:
                        self.env.reset()
                        continue

                f_returns.write(str(rewards) + "\n")
                if Debug:
                    print("[episode ", episode + 1, "] total rewards: ", rewards)

        with open(weight_out, 'w') as f_weight:
            f_weight.write(str(self.b) + "\n")
            # write the values of weights in row major order
            for i in range(self.W.shape[0]):
                for j in range(self.W.shape[1]):
                    f_weight.write(str(self.W[i][j]) + "\n")

        # visualization
        self.env.render()


if __name__ == '__main__':
    if len(sys.argv) != 9:
        print("The number of command parameters is incorrect.")
        exit(-1)

    mode = sys.argv[1]  # mode to run the environment in. Should be either "raw" or "tile"
    weight_out = sys.argv[2] # path to output the weights of the linear model
    returns_out = sys.argv[3] # path to output the returns of the agent
    episodes = int(sys.argv[4]) # the number of episodes your program should train the agent for
    max_iterations = int(sys.argv[5]) # the maximum of the length of an episode. (Terminate the current episode when it's reached)
    epsilon = int(sys.argv[6]) # the value ε for the epsilon-greedy strategy
    gamma = int(sys.argv[7]) # the discount factor γ.
    learning_rate = float(sys.argv[8]) # the learning rate α of the Q-learning algorithm

    # build and init 
    model = qlearning(mode, epsilon, gamma, learning_rate)

    # run and train the agent
    model.run(weight_out, returns_out, episodes, max_iterations)

