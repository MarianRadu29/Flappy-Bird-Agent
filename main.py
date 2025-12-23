import torch
import gymnasium as gym
import flappy_bird_gymnasium
from wrapper import Wrapper
from agent import Agent
from config import NUM_EPISODES

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

render_mode = "rgb_array"
env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)
env = Wrapper(env)
agent = Agent(env, device)
agent.train(NUM_EPISODES)
