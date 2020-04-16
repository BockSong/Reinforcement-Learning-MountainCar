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

class qlearning(object):
    def __init__(self, mode, epsilon, gamma, learning_rate):
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = learning_rate
        self.mode = mode
        if self.mode == "raw":
            self.state_space = 2
        elif self.mode == "tile":
            self.state_space = 2048
        else:
            print("Error mode.")
        self.action_space = 3
        self.env = MountainCar(mode)

        self.state = np.zeros((self.state_space))
        self.q = np.zeros((self.action_space))
        self.W = np.zeros((self.state_space, self.action_space))
        self.b = 0

    # given the current state and action, approximate thee action value
    def linear_approx(self, state):
        return np.dot(state.T, self.W).T + self.b

    # choose an action based on epsilon-greedy method
    def select_action(self):
        if np.random.rand() < self.epsilon:
            # selects uniformly at random from one of the 3 actions (0, 1, 2) with probability ε
            return np.random.randint(0, self.action_space + 1)
        else:
            # selects the optimal action with probability 1 − ε
            # In case of multiple maximum values, return the first one
            return np.argmax(self.linear_approx(self.state))

    def run(self, weight_out, returns_out, episodes, max_iterations):
        with open(returns_out, 'w') as f_returns:
            # perform training
            for episode in range(episodes):
                rewards = 0
                for i in range(max_iterations):
                    # call step
                    action = self.select_action()
                    self.state, reward, done = self.env.step(action)

                    # update parameters
                    if self.mode == "raw":
                        delta = self.state
                        self.W = self.W - self.lr * (self.q - (reward + self.gamma * np.max(self.linear_approx(self.state)))) * delta
                    elif self.mode == "tile":
                        # TODO:
                        pass
                    else:
                        print("Error mode.")

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

