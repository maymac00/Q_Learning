from abc import ABC, abstractmethod
import numpy as np
import warnings


class Callback(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.qlearn = None

    """
    Method to overload in case you need self.qlearn for the constructor. This is needed since you do not have 
    access to qlearn instance on Callback.__init__"""

    def initiate(self):
        pass


class ActionCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def onAction(self):
        pass


class UpdateCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def onUpdate(self):
        pass


class EpisodeCallback(Callback):

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def onEpisode(self):
        pass


class LearningRateDecay(EpisodeCallback):

    def __init__(self, decay_rate, period=10, min_lr=0.0001):
        super().__init__()
        self.decay = decay_rate
        self.period = period
        self.h = 0
        self.cont = 0
        self.min_lr = min_lr

    def onEpisode(self):
        if self.cont % self.period == 0:
            self.qlearn.alpha = max(self.qlearn.alpha * self.decay ** self.h, self.min_lr)
            self.h += 1

        self.cont += 1


class ActionCount(ActionCallback):
    def __init__(self):
        super().__init__()
        self.action_log = None

    def initiate(self):
        self.action_log = np.zeros((np.prod(self.qlearn.state_shape), self.qlearn.action_space), dtype="int")

    def onAction(self):
        self.action_log[self.qlearn.last_state, self.qlearn.last_action] += 1


class ConvergenceRate(UpdateCallback):
    def __init__(self):
        super().__init__()
        self.gradient_log = []
        self.alpha_log = []

    def onUpdate(self):
        self.gradient_log.append(self.qlearn.last_gradient)
        self.alpha_log.append(self.qlearn.alpha)
        pass

    def getUpdateValue(self):
        return np.array(self.gradient_log)*np.array(self.alpha_log)


class EpisodeRewardTracker(EpisodeCallback, UpdateCallback):
    def onUpdate(self):
        self.current_episode.append(self.qlearn.last_reward)

    def __init__(self):
        super().__init__()
        self.episodes_rewards = []
        self.current_episode = []

    def onEpisode(self):
        if len(self.episodes_rewards) == 0 and len(self.current_episode) == 0:
            return
        self.episodes_rewards.append(self.current_episode)
        self.current_episode = []

    def getRTforEpisode(self):
        return np.array([sum(ep) for ep in self.episodes_rewards])
