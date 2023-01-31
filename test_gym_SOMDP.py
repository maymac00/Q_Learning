import gym
import Q_Learning
from ExplorationSchedule import Epsilon, Bolzmann
from IPython.display import clear_output
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make("Taxi-v3")

    q = Q_Learning.QLearning((env.observation_space.n, 2), env.action_space.n, init="rand",
                             exploration_schedule=Bolzmann(0.9, 1000, 0.95, 15))
    # q = Q_Learning.QLearning((env.observation_space.n, 2), env.action_space.n,
    #                          exploration_schedule=Epsilon(0.9))
    historic_rt = []
    for i in range(1, 5000):
        state = env.reset()[0]

        epochs, penalties, reward, = 0, 0, 0
        done = False
        Rt = 0
        while not done:
            action = q.getAction(state)

            next_state, reward, done, _, info = env.step(action)
            # next_state = next_state[0]
            q.update_table(state, action, reward, next_state)

            state = next_state
            epochs += 1
            Rt += reward
        # print(f"Episode: {i} Rt: {Rt}")
        historic_rt.append(Rt)
        if i % 1000 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    plt.plot(historic_rt)
    plt.show()

    q.exportTable("Q0_policy")
    q2 = Q_Learning.QLearning((env.observation_space.n, 2), env.action_space.n).importTable("Q0_policy.npy")
    print("Training finished.\n")
