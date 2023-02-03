from abc import ABC, abstractmethod
import numpy as np


class IQTable(ABC):
    @abstractmethod
    def __getitem__(self, item):
        pass


class QTable(IQTable):
    def __init__(self, state_shape, action_space, reward_space=None, init="zeros"):
        if init == "zeros":
            self.state_shape = np.array(state_shape)
            self.action_space = np.array(action_space)
            self.reward_space = reward_space
            self.table = np.zeros((np.prod(self.state_shape), action_space, len(reward_space)))

        if init == "rand":
            self.state_shape = np.array(state_shape)
            self.action_space = np.array(action_space)
            self.reward_space = reward_space
            self.table = np.random.rand(np.prod(self.state_shape), action_space, len(reward_space))

    @classmethod
    def fromfile(cls, filename, state_shape, action_space):
        instance = cls(state_shape, action_space)
        instance.table = np.load(filename)
        if instance.table.shape != np.zeros((np.prod(state_shape), action_space), dtype="float32"):
            raise ValueError("Shapes of table and environment not consistent")
        return instance

    def getSingleObjective(self, item):
        if self.table.shape[2] > 1:
            if type(item) == tuple or type(item) == np.ndarray:
                return self.reward_space.convert(self.table[self.tuple_to_scalar_state(item)])
            elif type(item) == int:
                return self.reward_space.convert(self.table[item])
            else:
                raise TypeError("Table not accessible with type " + type(item).__name__)
        else:
            if type(item) == tuple or type(item) == np.ndarray:
                return self.table[self.tuple_to_scalar_state(item)]
            elif type(item) == int:
                return self.table[item]
            else:
                raise TypeError("Table not accessible with type " + type(item).__name__)

    def tuple_to_scalar_state(self, tup):
        if type(tup) == int:
            return int(tup)
        return np.ravel_multi_index(tup, self.state_shape)

    def export(self, filename):
        np.save(filename, self.table)

    def __getitem__(self, item):
        if type(item) == tuple or type(item) == np.ndarray:
            return self.table[self.tuple_to_scalar_state(item)]
        elif type(item) == int:
            return self.table[item]
        else:
            raise TypeError("Table not accessible with type " + type(item).__name__)

