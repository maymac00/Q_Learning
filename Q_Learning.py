import numpy as np
from QTables import QTable
from Callbacks import Callback, UpdateCallback, ActionCallback, EpisodeCallback
from ExplorationSchedule import ExplorationSchedule
from RewardSpace import RewardSpace
from typing import Type


# TODO: Poder restringir el nombre d'accions i traduir les per a que no hagi de ser acció == index
# TODO: Modificar callbakcs perque permetin condicions de parada

class QLearning:

    def __init__(self, state_shape, action_space, discount_factor=0.9, learning_rate=0.1, init="zeros",
                 exploration_schedule: Type[ExplorationSchedule] = None,
                 reward_space: Type[RewardSpace] = None
                 ):
        self.callbacks = []
        self.state_shape = state_shape
        self.action_space = action_space
        self.qtable = QTable(state_shape, action_space, reward_space=reward_space, init=init)
        self.gamma = discount_factor
        self.alpha = learning_rate
        self.exploration_schedule = exploration_schedule
        self.last_action = None
        self.last_state = None
        self.last_gradient = None
        self.last_reward = None

    def addCallbacks(self, callbacks: list = []):
        for c in callbacks:
            if not issubclass(type(c), Callback):
                raise TypeError("Element of class ", type(c).__name__, " not a subclass from Callback")
            c.qlearn = self
            c.initiate()
        self.callbacks = callbacks

    def update_table(self, s, a, r, s_prime):
        expected_return_ind = np.argmax(self.qtable[s_prime])
        gradient = (
                r + self.gamma * self.qtable.table[s_prime][expected_return_ind] - self.qtable[s][a])
        self.qtable.table[s][a] += self.alpha * gradient
        self.last_gradient = gradient
        self.last_reward = r
        for c in self.callbacks:
            if issubclass(type(c), UpdateCallback):
                c.onUpdate()

    def getAction(self, s):
        if self.exploration_schedule is None:
            act = np.argmax(self.qtable[s])
        else:
            act = self.exploration_schedule.chooseAction(self.qtable[s])
        self.last_action = act
        self.last_state = s

        for c in self.callbacks:
            if issubclass(type(c), ActionCallback):
                c.onAction()
        return act

    def exportTable(self, filename):
        self.qtable.export(filename)

    def importTable(self, filename):
        self.qtable = QTable.fromfile(filename, self.qtable.state_shape, self.qtable.action_space)

    def newEpisode(self):
        for c in self.callbacks:
            if issubclass(type(c), EpisodeCallback):
                c.onEpisode()
