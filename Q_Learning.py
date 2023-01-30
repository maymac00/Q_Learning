import numpy as np


class QTable:
    def __init__(self, state_shape, action_space, init="zeros"):
        if init == "zeros":
            self.state_shape = np.array(state_shape)
            self.action_space = np.array(action_space)
            self.table = np.zeros((np.prod(self.state_shape), action_space), dtype="float32")
        if init == "rand":
            self.state_shape = np.array(state_shape)
            self.action_space = np.array(action_space)
            self.table = np.random.rand(np.prod(self.state_shape), action_space)

    @classmethod
    def fromfile(cls, filename, state_shape, action_space):
        instance = cls(state_shape, action_space)
        instance.table = np.load(filename)
        if instance.table.shape != np.zeros((np.prod(state_shape), action_space), dtype="float32"):
            raise ValueError("Shapes of table and environment not consistent")
        return instance

    def tuple_to_scalar_state(self, tup):
        if type(tup) == int:
            return int(tup)
        return np.ravel_multi_index(tup, self.state_shape)

    def export(self, filename):
        np.save(filename, self.table)

    def __getitem__(self, item):
        if type(item) == tuple:
            return self.table[self.tuple_to_scalar_state(item)]
        elif type(item) == int:
            return self.table[item]
        else:
            raise TypeError("Table not accessible with type " + type(item).__name__)


class QLearning:

    def __init__(self, state_shape, action_space, discount_factor=0.9, learning_rate=0.1, init="zeros",
                 exploration_schedule=None):
        self.qtable = QTable(state_shape, action_space, init=init)
        self.gamma = discount_factor
        self.alpha = learning_rate
        self.exploration_schedule = exploration_schedule

    def update_table(self, s, a, r, s_prime):
        self.qtable[s][a] += self.alpha * (r + self.gamma * self.qtable[s_prime].max() - self.qtable[s][a])

    def getAction(self, s):
        if self.exploration_schedule is None:
            return np.argmax(self.qtable[s])
        else:
            return self.exploration_schedule.chooseAction(self.qtable[s])

    def exportTable(self, filename):
        self.qtable.export(filename)

    def importTable(self, filename):
        self.qtable = QTable.fromfile(filename, self.qtable.state_shape, self.qtable.action_space)
