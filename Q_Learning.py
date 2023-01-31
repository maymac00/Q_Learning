import numpy as np
from QTables import QTable


class QLearning:

    def __init__(self, state_shape, action_space, discount_factor=0.9, learning_rate=0.1, init="zeros",
                 exploration_schedule=None, reward_space=None):
        self.qtable = QTable(state_shape, action_space, reward_space=reward_space, init=init)
        self.gamma = discount_factor
        self.alpha = learning_rate
        self.exploration_schedule = exploration_schedule

    def update_table(self, s, a, r, s_prime):
        expected_return_ind = np.argmax(self.qtable[s_prime])
        self.qtable.table[s][a] += self.alpha * (r + self.gamma * self.qtable.table[s_prime][expected_return_ind] - self.qtable[s][a])

    def getAction(self, s):
        if self.exploration_schedule is None:
            return np.argmax(self.qtable[s])
        else:
            return self.exploration_schedule.chooseAction(self.qtable[s])

    def exportTable(self, filename):
        self.qtable.export(filename)

    def importTable(self, filename):
        self.qtable = QTable.fromfile(filename, self.qtable.state_shape, self.qtable.action_space)
