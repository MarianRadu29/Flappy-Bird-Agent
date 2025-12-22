import matplotlib
import matplotlib.pyplot as plt
import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class Plotter:
    def __init__(self):
        self.episode_rewards = []

        plt.ion()

    def update(self, reward):
        self.episode_rewards.append(reward)

        plt.figure(1)
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)

        plt.clf()
        plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.plot(rewards_t.numpy())

        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)

        if is_ipython:
            display.display(plt.gcf())
            display.clear_output(wait=True)

    def close(self):
        plt.ioff()
        plt.savefig("training_plot.png")
        plt.show()
