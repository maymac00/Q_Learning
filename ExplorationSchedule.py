from abc import ABC, abstractmethod
import numpy as np


class ExplorationSchedule(ABC):
    log = []  # 1 exp 0 greed

    @abstractmethod
    def chooseAction(self, qs, **kwargs):
        pass


class Epsilon(ExplorationSchedule):
    def __init__(self, startp, decay=0.999):
        self.p0 = startp
        self.decay = decay
        self.count = 0

    def chooseAction(self, qs, **kwargs):
        p = self.p0 * self.decay ** self.count
        self.count += 1
        if np.random.rand() < p:
            ExplorationSchedule.log.append(1)
            return np.random.choice([i for i in range(qs.shape[0])])
        else:
            ExplorationSchedule.log.append(0)
            return np.argmax(qs)


class Bolzmann(ExplorationSchedule):
    def __init__(self, p, T0, alpha, L):
        self.T0 = T0
        self.alpha = alpha
        self.L = L
        self.h = 0
        self.T = T0
        self.count = 0
        self.t_min = 0.001

    def chooseAction(self, qs, **kwargs):
        if self.count % self.L == 0:
            self.h += 1
            self.T = self.T0 * self.alpha ** self.h
        self.count += 1
        if self.T < self.t_min:
            return np.argmax(qs)
        s = np.exp(qs / self.T).sum()
        if s == 0.0 or s == np.inf:
            ExplorationSchedule.log.append(0)
            return np.argmax(qs)
        else:
            probs = np.array([np.exp(qsa / self.T) / s for qsa in qs], dtype=np.float64)
            probs = np.round(probs, 4)
            if probs.sum() != 1.0:
                if probs.sum() == 0.0:
                    ExplorationSchedule.log.append(0)
                    return np.argmax(qs)
                probs /= probs.sum()
            ExplorationSchedule.log.append(1)
            return np.random.choice([i for i in range(qs.shape[0])], p=probs)
