import gym
import Q_Learning
from ExplorationSchedule import Epsilon, Bolzmann
from RewardSpace import Scalarization
from IPython.display import clear_output
import matplotlib.pyplot as plt

if __name__ == '__main__':
    q = Q_Learning.QLearning(400, 6, init="rand",
                             reward_space=Scalarization([0.75, 0.25]),
                             exploration_schedule=Bolzmann(0.9, 1000, 0.95, 15))

    q.update_table(200, 4, [3, -1], 201)
    q.getAction(201)
