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

# finite difference approximation
'''
if Debug:
    grad = np.zeros(self.W.shape[0])
    for m in range(1, len(grad) + 1):
        d = np.zeros(self.W.shape[0])
        d[m] = 1
        v = np.dot(self.x, self.W + epsilon * d) + self.b
        v -= np.dot(self.x, self.W - epsilon * d) + self.b
        v /= 2 * epsilon
        grad[m] = v
    
    if np.linalg.norm(dW - grad) > diff_th:
        print("Gradient compute error!")
        print("dW: ", dW)
        print("grad: ", grad)
'''

class rl(object):
    def __init__(self, mode, epsilon, gamma, learning_rate):
        self.mode = mode
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate

    # SGD_step: update params by taking one SGD step
    # <x> a 1-D numpy array
    # <y> an integer within [0, num_class - 1]
    def SGD_step(self, x, y):
        # perform forward propogation and compute intermediate results
        for layer in self.layers:
            x = layer.forward(x)
        loss, _ = self.criterion.forward(x, y)

        # perform back propagation and update parameters
        delta = self.criterion.backward()
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)
            
        return loss


    def train(self, episodes, max_iterations):
        with open(metrics_out, 'w') as f_metrics:
            # perform training
            for epoch in range(episodes):
                loss = 0
                for idx in range(len(dataset)):
                    loss = self.SGD_step(dataset[idx][1], dataset[idx][0])
                    if Debug and (idx % 1000 == 0):
                        print("[Epoch ", epoch + 1, "] Step ", idx + 1, ", current_loss: ", loss)

                train_loss, train_error = self.evaluate(train_input, train_out)
                test_loss, test_error = self.evaluate(test_input, test_out)

                if Debug:
                    print("[Epoch ", epoch + 1, "] ", end='')
                    print("train_loss: ", train_loss, end=' ')
                    print("train_error: ", train_error)
                    print("test_loss: ", test_loss, end=' ')
                    print("test_error: ", test_error)

                f_metrics.write("epoch=" + str(epoch) + " crossentryopy(train): " + str(train_loss) + "\n")
                f_metrics.write("epoch=" + str(epoch) + " crossentryopy(test): " + str(test_loss) + "\n")

    # predict y given an array x
    # not used
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        sm = np.exp(x) / np.sum(np.exp(x), axis=0)
        return np.argmax(sm)

    def evaluate(self, in_path, out_path, write = False):
        total_loss, error, total = 0., 0., 0.

        with open(in_path, 'r') as f_in:
            with open(out_path, 'a') as f_out:
                for line in f_in:
                    split_line = line.strip().split(',')
                    y = int(split_line[0])
                    x = np.asarray(split_line[1:], dtype=int)

                    for layer in self.layers:
                        x = layer.forward(x)
                    loss, pred = self.criterion.forward(x, y)

                    total_loss += loss
                    if pred != y:
                        error += 1
                    if write:
                        f_out.write(str(pred) + "\n")
                    total += 1

        return total_loss / total, error / total


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
    model = rl(mode, epsilon, gamma, learning_rate)

    # training
    model.train(episodes, max_iterations)

    # testing: evaluate and write labels to output files
    model.evaluate(weight_out, returns_out, True)

