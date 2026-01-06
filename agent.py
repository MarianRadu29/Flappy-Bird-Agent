import math
import os
import random

from monitor import Monitor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN
from replay_memory import ReplayMemory, Transition
from config import MEMORY_SIZE, BATCH_SIZE, GAMMA, LR, EPS_START, EPS_END, EPS_DECAY, TAU, MODEL_PATH, SAVE_EVERY, PERIODIC_MODEL_PATH


class Agent:
    def __init__(self, env, device):
        self.env = env
        self.device = device

        # Env info
        state, _ = env.reset()
        self.n_actions = env.action_space.n
        self.n_frames = state.shape[0]  # (n_frames, 84, 84)

        # Networks
        self.policy_net = DQN(self.n_frames, self.n_actions).to(device)
        self.target_net = DQN(self.n_frames, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)

        self.steps_done = 0

        self.monitor = Monitor()
        self.best_score = -float("inf")

        self._load_checkpoint()

    def _load_checkpoint(self):
        if not os.path.exists(MODEL_PATH):
            return

        checkpoint = torch.load(MODEL_PATH, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.steps_done = checkpoint.get("steps_done", 0)
        print(f"Loaded checkpoint from {MODEL_PATH} at step {self.steps_done}")

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path=MODEL_PATH):
        torch.save(
            {
                "model_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps_done": self.steps_done
            },
            path
        )

    def act(self, state):
        """state = numpy array (n_frames, 84, 84)"""
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(
            -self.steps_done / EPS_DECAY
        )
        self.steps_done += 1
        if eps == 0.05:
            print(f"Epsilon: {eps:.4f}")

        if random.random() > eps:
            with torch.no_grad():
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                return self.policy_net(state_tensor).max(1).indices.item()

        return self.env.action_space.sample()

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            device=self.device,
            dtype=torch.bool
        )

        non_final_next_states = torch.tensor(
            np.array([s for s in batch.next_state if s is not None]),
            dtype=torch.float32,
            device=self.device
        )

        state_batch = torch.tensor(
            np.array(batch.state),
            dtype=torch.float32,
            device=self.device
        )
        action_batch = torch.tensor(
            np.array(batch.action),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            np.array(batch.reward),
            dtype=torch.float32,
            device=self.device
        )
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        # Double DQN
        with torch.no_grad():
            # selectez actiunea optima folosind Policy Net (argmax Q_policy)
            best_actions = self.policy_net(non_final_next_states).max(1).indices.unsqueeze(1)

            # evaluez valoarea acelei actiuni folosind Target Net (Q_target)
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).gather(1, best_actions).squeeze(1)
            )

        expected_q_values = reward_batch + GAMMA * next_state_values

        loss = nn.MSELoss()(
            state_action_values,
            expected_q_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # Soft update target net
        with torch.no_grad():
            for key in self.target_net.state_dict():
                self.target_net.state_dict()[key].copy_(
                    self.policy_net.state_dict()[key] * TAU +
                    self.target_net.state_dict()[key] * (1.0 - TAU)
                )

    def train(self, num_episodes):
        self.monitor.log(f"Training on {self.device}")

        try:
            for episode in range(num_episodes):
                obs, _ = self.env.reset()
                # obs ramane numpy array pe CPU!
                state = obs  # shape: (n_frames, 84, 84)

                self.policy_net.train()
                episode_reward = 0

                while True:
                    action = self.act(state)  # return int

                    obs, reward, terminated, truncated, info = self.env.step(action)
                    episode_reward += reward

                    done = terminated or truncated
                    if done:
                        next_state = None
                    else:
                        next_state = obs  # numpy array pe CPU

                    # Buffer pe CPU: numpy, int, numpy/None, float
                    self.memory.push(state, action, next_state, reward)

                    state = next_state

                    self.learn()

                    if done:
                        score = info.get("score", 0)

                        self.monitor.update_plot(episode_reward)

                        if score > self.best_score:
                            self.best_score = score
                            self.monitor.log(
                                f"New high score | "
                                f"Episode {episode} | "
                                f"Score {score} | "
                                f"Reward {episode_reward:.2f}"
                            )

                        if (episode + 1) % SAVE_EVERY == 0:
                            save_path = f"{PERIODIC_MODEL_PATH}/model_ep_{episode + 1}.pth"
                            self.save(save_path)

                        break

        except KeyboardInterrupt:
            self.monitor.log("Training interrupted")

        finally:
            self.monitor.log("Saving model...")
            self.save()
            self.monitor.close()
            self.env.close()
