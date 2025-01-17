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
        self.env = MountainCar(mode)
        self.state_space = self.env.state_space
        self.action_space = 3

        self.W = np.zeros((self.state_space, self.action_space))
        self.b = 0

    # given the current state and action, approximate thee action value (q_s)
    def linear_approx(self, state):
        #return np.dot(state.T, self.W).T + self.b
        return state.dot(self.W) + self.b

    # choose an action based on epsilon-greedy method
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            # selects uniformly at random from one of the 3 actions (0, 1, 2) with probability ε
            return np.random.randint(0, self.action_space)
        else:
            # selects the optimal action with probability 1 − ε
            # In case of multiple maximum values, return the first one
            return np.argmax(self.linear_approx(state))

    def transfer_state(self, state):
        if self.mode == "raw":
            return np.fromiter(state.values(), dtype=float)
        elif self.mode == "tile":
            idx = sorted(state.keys())
            trans_state = np.zeros((self.state_space))
            trans_state[idx] = 1
            return trans_state
        else:
            print("Error mode.")
            return

    def run(self, weight_out, returns_out, episodes, max_iterations):
        with open(returns_out, 'w') as f_returns:
            # perform training
            for episode in range(episodes):
                rewards = 0
                state = self.transfer_state(self.env.reset())
                if Debug:
                    print("episode " + str(episode) + " init state: ", end = "")
                    print(state)
                for i in range(max_iterations):
                    # call step
                    action = self.select_action(state)
                    next_state, reward, done = self.env.step(action)
                    next_state = self.transfer_state(next_state)

                    if Debug and i % 100 == 0:
                        print("episode " + str(episode) + " iter " + str(i) + ", action: " + str(action)
                                                                            + " next state: ", end = "")
                        print(next_state)

                    # update w_a
                    delta = state
                    cur_q = self.linear_approx(state)
                    next_q = self.linear_approx(next_state)
                    self.W[:, action] = self.W[:, action] - self.lr * (cur_q[action] - 
                                      (reward + self.gamma * np.max(next_q))) * delta
                    # update bias
                    self.b = self.b - self.lr * (cur_q[action] - 
                                      (reward + self.gamma * np.max(next_q)))

                    state = next_state
                    rewards += reward
                    if done:
                        break

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
        # self.env.render()

    def close(self):
        self.env.close()


if __name__ == '__main__':
    if len(sys.argv) != 9:
        print("The number of command parameters is incorrect.")
        exit(-1)

    mode = sys.argv[1]  # mode to run the environment in. Should be either "raw" or "tile"
    weight_out = sys.argv[2] # path to output the weights of the linear model
    returns_out = sys.argv[3] # path to output the returns of the agent
    episodes = int(sys.argv[4]) # the number of episodes your program should train the agent for
    max_iterations = int(sys.argv[5]) # the maximum of the length of an episode
    epsilon = float(sys.argv[6]) # the value ε for the epsilon-greedy strategy
    gamma = float(sys.argv[7]) # the discount factor γ.
    learning_rate = float(sys.argv[8]) # the learning rate α of the Q-learning algorithm

    # build and init 
    model = qlearning(mode, epsilon, gamma, learning_rate)

    # train the agent
    model.run(weight_out, returns_out, episodes, max_iterations)

    #model.close()
