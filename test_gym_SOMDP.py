import gym
import Q_Learning
from ExplorationSchedule import Epsilon, Bolzmann
from IPython.display import clear_output
import matplotlib.pyplot as plt
from Callbacks import LearningRateDecay, ActionCount, ConvergenceRate, EpisodeRewardTracker

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make("Taxi-v3")

    q = Q_Learning.QLearning(env.observation_space.n, env.action_space.n, init="rand",
                             exploration_schedule=Bolzmann(0.9, 1000, 0.95, 15), learning_rate=0.7)

    c = ConvergenceRate()
    historic_RT = EpisodeRewardTracker()
    q.addCallbacks([
             LearningRateDecay(0.999, period=25),
             ActionCount(),
             c,
             historic_RT
             ])
    # q = Q_Learning.QLearning((env.observation_space.n, 2), env.action_space.n,
    #                          exploration_schedule=Epsilon(0.9))

    for i in range(1, 5000):
        state = env.reset()[0]
        q.newEpisode()
        epochs = 0
        done = False
        Rt = 0
        while not done:
            action = q.getAction(state)

            next_state, reward, done, _, info = env.step(action)
            # next_state = next_state[0]
            q.update_table(state, action, reward, next_state)

            state = next_state

            Rt += reward
        # print(f"Episode: {i} Rt: {Rt}")
        epochs += 1
        if i % 1000 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    plt.plot(c.getUpdateValue())
    plt.show()
    plt.plot(historic_RT.getRTforEpisode())
    plt.show()

    q.exportTable("Q0_policy")
    print("Training finished.\n")
