import os
import logging
import datetime
import matplotlib
import matplotlib.pyplot as plt
import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class Monitor:
    def __init__(self):
        self.episode_rewards = []
        self._setup_logger()
        self._setup_plotter()

    def _setup_logger(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/training_{timestamp}.txt"

        self.logger = logging.getLogger("FlappyBirdAgent")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console_handler)

    def _setup_plotter(self):
        plt.ion()

    def log(self, message):
        self.logger.info(message)

    def update_plot(self, reward):
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
        plt.savefig("training_plot.png.png")
        plt.show()
