from abc import ABC, abstractmethod
import numpy as np


class RewardSpace(ABC):
    @abstractmethod
    def convert(self, arr):
        pass

    @abstractmethod
    def __len__(self):
        pass


class Scalarization(RewardSpace):

    def __init__(self, weights):
        self.weights = np.array(weights)
        self.weights = np.expand_dims(weights, axis=1)

    def __len__(self):
        return self.weights.shape[0]

    def convert(self, arr):
        return np.dot(arr, self.weights)
