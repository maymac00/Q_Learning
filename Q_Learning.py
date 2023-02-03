import numpy as np
from QTables import QTable
from Callbacks import Callback, UpdateCallback, ActionCallback, EpisodeCallback
from ExplorationSchedule import ExplorationSchedule
from RewardSpace import RewardSpace, SingleObjective
from typing import Type


# TODO: ActionMap encapsulat en una classe per evitar fer if else
# TODO: Modificar callbakcs perque permetin condicions de parada

class QLearning:

    def __init__(self, state_shape, action_space, discount_factor=0.9, learning_rate=0.1, init="zeros",
                 action_map: list = None,
                 exploration_schedule: Type[ExplorationSchedule] = None,
                 reward_space: Type[RewardSpace] = SingleObjective()
                 ):
        self.callbacks = []
        self.state_shape = state_shape
        self.action_space = action_space
        self.action_map = action_map
        if action_map is not None:
            if len(action_map) != action_space:
                raise ValueError("Action map attribute must have len == action_space")
        else:
            self.action_map = [i for i in range(self.action_space)]

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

    def __greedy_act__(self, s):
        act = np.argmax(self.qtable.getSingleObjective(s))
        return act

    def update_table(self, s, a, r, s_prime):
        a = self.action_map.index(a)
        expected_return_ind = self.__greedy_act__(s)
        gradient = (
                r + self.gamma * self.qtable[s_prime][expected_return_ind])
        self.qtable[s][a] = self.qtable[s][a]*(1-self.alpha) + self.alpha*gradient
        self.last_gradient = gradient
        self.last_reward = r

        for c in self.callbacks:
            if issubclass(type(c), UpdateCallback):
                c.onUpdate()

    def getAction(self, s):
        if self.exploration_schedule is None:
            act = self.__greedy_act__(s)
        else:
            act = self.exploration_schedule.chooseAction(self.qtable.getSingleObjective(s))
        self.last_action = act
        self.last_state = s

        for c in self.callbacks:
            if issubclass(type(c), ActionCallback):
                c.onAction()

        return self.action_map[act]

    def exportTable(self, filename):
        self.qtable.export(filename)

    def importTable(self, filename):
        self.qtable = QTable.fromfile(filename, self.qtable.state_shape, self.qtable.action_space)

    def newEpisode(self):
        for c in self.callbacks:
            if issubclass(type(c), EpisodeCallback):
                c.onEpisode()
